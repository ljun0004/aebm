import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time
import hashlib


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        learningrate, warmup = lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():

            if args.vae_mode == "kl":
                # VAE encode
                samples = samples.to(device, non_blocking=True)
                if args.use_cached:
                    moments = samples
                    posterior = DiagonalGaussianDistribution(moments)
                else:
                    posterior = vae.encode(samples)
                # normalize the std of latent to be 1. Change it if you use a different tokenizer
                x = posterior.sample().mul_(0.2325)
                cookbook = None
                gt_indices = None

            elif args.vae_mode == "vq":
                # VQGAN encode
                if args.use_cached:
                    x, gt_indices = samples
                    x = x.to(device, non_blocking=True)
                    gt_indices = gt_indices.to(device, non_blocking=True)
                else:
                    samples = samples.to(device, non_blocking=True)
                    x, _, _, info = vae.encode(samples)
                    _, _, token_indices = info
                    gt_indices = token_indices.clone().long()
                cookbook = vae.quantize.embedding.weight

            else:
                raise NotImplementedError

        # forward
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            loss, ddpmloss, celoss, reloss, logitsnorm, qnorm, pimax, scorenorm, tembnorm, scale = model(x, labels, gt_indices=gt_indices, cookbook=cookbook, warmup=warmup)

        loss_value = loss.item()
        ddpmloss_value = ddpmloss.item()
        celoss_value = celoss.item()
        reloss_value = reloss.item()
        logitsnorm_value = logitsnorm.item()
        qnorm_value = qnorm.item()
        pimax_value = pimax.item()
        scorenorm_value = scorenorm.item()
        tembnorm_value = tembnorm.item()
        scale_value = scale.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        gradnorm = loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        gradnorm_value = gradnorm.item()

        # with torch.no_grad():
        #     mar = model.module if hasattr(model, "module") else model
        #     mar.final_layer.proj.weight_g.clamp_(0.1, 1.0)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value, ddpm=ddpmloss_value, ce=celoss_value, re=reloss_value, grad=gradnorm_value, logits=logitsnorm_value, q=qnorm_value, pi=pimax_value, score=scorenorm_value, temb=tembnorm_value, scale=scale_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        ddpmloss_value_reduce = misc.all_reduce_mean(ddpmloss_value)
        celoss_value_reduce = misc.all_reduce_mean(celoss_value)

        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('ddpm_loss', ddpmloss_value_reduce, epoch_1000x)
            log_writer.add_scalar('ce_loss', celoss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):

    if args.vae_mode == "kl":
        cookbook = None
    elif args.vae_mode == "vq":
        cookbook = vae.quantize.embedding.weight
    else:
        raise NotImplementedError

    model_without_ddp.eval()
    num_steps = args.eval_num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}-epoch{}-{}".format(args.num_iter,
                                                                                                                args.num_sampling_steps,
                                                                                                                args.temperature,
                                                                                                                args.cfg_schedule,
                                                                                                                cfg,
                                                                                                                args.eval_num_images,
                                                                                                                epoch,
                                                                                                                args.sampling_mode))

    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate or args.online_eval:
        save_folder = save_folder + "_eval"
    print("Save to:", save_folder)

    if not os.path.exists(save_folder):
        
        if misc.get_rank() == 0:
                os.makedirs(save_folder)
        torch.distributed.barrier()

        # switch to ema params
        if use_ema:
            model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
                assert name in ema_state_dict
                ema_state_dict[name] = ema_params[i]
            print("Switch to ema")
            model_without_ddp.load_state_dict(ema_state_dict)

        class_num = args.class_num
        assert args.eval_num_images % class_num == 0  # number of images per class must be the same
        class_label_gen_world = np.arange(0, class_num).repeat(args.eval_num_images // class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
        world_size = misc.get_world_size()
        local_rank = misc.get_rank()
        used_time = 0
        gen_img_cnt = 0

        for i in range(num_steps):
            print("Generation step {}/{}".format(i, num_steps))

            labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                    world_size * batch_size * i + (local_rank + 1) * batch_size]
            labels_gen = torch.Tensor(labels_gen).long().cuda()

            torch.cuda.synchronize()
            start_time = time.time()

            # generation
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    sampled_tokens = model_without_ddp.sample_tokens(eval_bsz=batch_size, cookbook=cookbook, num_iter=args.num_iter, cfg=cfg,
                                                                        cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                        temperature=args.temperature)                       

                    if args.vae_mode == "kl":
                        sampled_images = vae.decode(sampled_tokens / 0.2325)
                    elif args.vae_mode == "vq":
                        sampled_images = vae.decode(sampled_tokens)
                    else:
                        raise NotImplementedError

            # measure speed after the first generation batch
            if i >= 1:
                torch.cuda.synchronize()
                used_time += time.time() - start_time
                gen_img_cnt += batch_size
                print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

            torch.distributed.barrier()
            sampled_images = sampled_images.detach().cpu().float()
            sampled_images = (sampled_images + 1) / 2

            # distributed save
            for b_id in range(sampled_images.size(0)):
                img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= args.eval_num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

        torch.distributed.barrier()
        time.sleep(10)

        # back to no ema
        if use_ema:
            print("Switch back from ema")
            model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if misc.get_rank() == 0:
        if log_writer is not None:
            if args.img_size == 256:
                input2 = None
                fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
            else:
                raise NotImplementedError
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=save_folder,
                input2=input2,
                fid_statistics_file=fid_statistics_file,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            fid = metrics_dict['frechet_inception_distance']
            inception_score = metrics_dict['inception_score_mean']
            postfix = ""
            if use_ema:
                postfix = postfix + "_ema"
            if not cfg == 1.0:
                postfix = postfix + "_cfg{}".format(cfg)
            log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
            log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
            print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
            # remove temporal saving folder
            shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)


def unconditional_generate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True, data_loader=None):

    if data_loader is not None:
        samples, labels = next(iter(data_loader))
        device = torch.device("cuda")

        labels = labels.to(device, non_blocking=True)

        if args.vae_mode == "kl":
            # VAE encode
            samples = samples.to(device, non_blocking=True)
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)
            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            h = posterior.mode().mul_(0.2325)
            cookbook = None
            gt_indices = None

        elif args.vae_mode == "vq":
            # VQGAN encode
            if args.use_cached:
                h, gt_indices = samples
                h = h.to(device, non_blocking=True)
                gt_indices = gt_indices.to(device, non_blocking=True)
            else:
                samples = samples.to(device, non_blocking=True)
                h, _, _, info = vae.encode(samples)
                _, _, token_indices = info
                gt_indices = token_indices.clone().long()
            cookbook = vae.quantize.embedding.weight

        else:
            raise NotImplementedError

        # x = samples.detach().cpu().contiguous().numpy().tobytes()
        # print("samples_md5:", hashlib.md5(x).hexdigest())
        # print(f"unconditional_generate: samples: {samples.shape, samples.min().item(), samples.max().item()}")
        # print(f"unconditional_generate: h: {h.shape, h.min().item(), h.max().item()}")
    
        idx = 0
        imgs = h[idx].unsqueeze(0)
        labels = labels[idx].unsqueeze(0)
    else:
        imgs = None
        labels = None

    assert args.gen_num_images >= misc.get_world_size()

    model_without_ddp.eval()
    num_steps = args.gen_num_images // (batch_size * misc.get_world_size())
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}-{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.gen_num_images,
                                                                                                     args.sampling_mode))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.generate or args.online_gen:
        save_folder = save_folder + "_gen"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            # print(f"Unconditional_generate: cookbook: {cookbook.shape}, gt_indices: {gt_indices.shape}")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                sampled_tokens = model_without_ddp.sample_tokens(eval_bsz=batch_size, cookbook=cookbook, num_iter=args.num_iter, 
                                                                cfg=cfg, cfg_schedule=args.cfg_schedule, temperature=args.temperature,
                                                                imgs=imgs, labels=labels, gt_indices=gt_indices, sampling_mode=args.sampling_mode)
                # sampled_tokens = h
                # print(f"unconditional_generate: sampled_tokens: {sampled_tokens.shape}")
                                                                
                if args.vae_mode == "kl":
                    sampled_images = vae.decode(sampled_tokens / 0.2325)
                elif args.vae_mode == "vq":
                    sampled_images = vae.decode(sampled_tokens)
                else:
                    raise NotImplementedError
                # sampled_images = samples
                # print(f"unconditional_generate: sampled_images: {sampled_images.shape}")

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu().float()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            print(f"unconditional_generate: img_id: {img_id}, args.gen_num_images: {args.gen_num_images}")
            if img_id >= args.gen_num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            filename = f"{epoch}_{str(img_id).zfill(5)}.png"
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, gen_img)
            print(f"unconditional_generate: save_path: {save_path}")

    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():

            if args.vae_mode == "kl":
                posterior = vae.encode(samples)
                moments = posterior.parameters
                posterior_flip = vae.encode(samples.flip(dims=[3]))
                moments_flip = posterior_flip.parameters

            elif args.vae_mode == "vq":
                h, _, _, info = vae.encode(samples)
                _, _, token_indices = info
                gt_indices = token_indices.reshape(h.size(0), -1).long()
                h_flip, _, _, info_flip = vae.encode(samples.flip(dims=[3]))
                _, _, token_indices_flip = info_flip
                gt_indices_flip = token_indices_flip.reshape(h_flip.size(0), -1).long()
            
            else:
                raise NotImplementedError

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if args.vae_mode == "kl":
                np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())
            elif args.vae_mode == "vq":
                np.savez(save_path, h=h[i].cpu().numpy(), gt_indices = gt_indices[i].cpu().numpy(), h_flip=h_flip[i].cpu().numpy(), gt_indices_flip = gt_indices_flip[i].cpu().numpy())
            else:
                raise NotImplementedError

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
