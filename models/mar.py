from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from einops import rearrange
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from timm.models.vision_transformer import Block, PatchEmbed, Attention, Mlp, LayerScale, DropPath, LayerNorm, RmsNorm
from torch.nn import functional as F
from torch.nn.utils import weight_norm
# import pytorch_lightning as L
from itertools import chain
from torch.nn.attention import sdpa_kernel, SDPBackend

from diffusion import create_diffusion
from models.ddpmloss import DDPMLoss

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 final_embed_dim=1024,
                 cookbook_size=16384,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=256,
                 mask_ratio_min=0.50,
                 mask_ratio_max=1.00,
                 mask_ratio_mu=0.75,
                 mask_ratio_std=0.25,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 grad_checkpointing=False,
                 adapter_embed_dim=1024, adapter_depth=3, adapter_num_heads=16, adapter_mlp_ratio=4,
                 num_sampling_steps='100', diffusion_batch_mul=1, 
                 celoss_scale=1.0, ddpmloss_scale=1.0, reloss_scale=1.0, 
                 min_logit_scale=1.0, max_logit_scale=1.0,
                 alpha=1.0, beta=1.0, 
                 encoder_adaln_mod=False, decoder_adaln_mod=False, final_layer_adaln_mod=False,
                 learn_sigma=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE, VQGAN and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        self.cookbook_size = cookbook_size

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.token_h = self.token_w = img_size // vae_stride
        self.token_len = self.token_h * self.token_w
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.z_proj_dim = final_embed_dim // patch_size**2
        self.out_channels = vae_embed_dim * 2 if learn_sigma else vae_embed_dim
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim, _freeze=True)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim), requires_grad=False)

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        # self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # Calculate Z-scores for boundaries
        lower_bound = (mask_ratio_min - mask_ratio_mu) / mask_ratio_std
        upper_bound = (mask_ratio_max - mask_ratio_mu) / mask_ratio_std

        # Create Generator
        self.mask_ratio_generator = stats.truncnorm(
            lower_bound, 
            upper_bound, 
            loc=mask_ratio_mu, 
            scale=mask_ratio_std
        )

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        # self.x_embedder = PatchEmbed(img_size // vae_stride, patch_size, vae_embed_dim, encoder_embed_dim, bias=True)

        # self.z_proj = nn.Conv2d(vae_embed_dim, self.z_proj_dim, kernel_size=1, bias=False)
        self.z_proj = nn.Linear(vae_embed_dim, self.z_proj_dim, bias=False)
        # self.z_proj_ln = nn.Identity()
        # self.z_proj_ln = nn.LayerNorm(self.z_proj_dim, elementwise_affine=False, eps=1e-5)
        # self.encoder_embed = nn.Linear(final_embed_dim, encoder_embed_dim, bias=True)
        # self.encoder_embed_ln = nn.LayerNorm(encoder_embed_dim, elementwise_affine=True, eps=1e-5)

        # self.buffer_size = buffer_size
        # self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            EBTBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
                     adaln_mod=encoder_adaln_mod) for _ in range(encoder_depth)])
        
        # if encoder_adaln_mod:
        #     self.encoder_norm = norm_layer(encoder_embed_dim, elementwise_affine=False, eps=1e-5)
        # else:
        #     self.encoder_norm = norm_layer(encoder_embed_dim, elementwise_affine=True, eps=1e-5)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        # self.xt_embed = nn.Linear(self.token_embed_dim, decoder_embed_dim, bias=True)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # self.decoder_embed_ln = nn.LayerNorm(decoder_embed_dim, eps=1e-5)

        # self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        # self.decoder_blocks = nn.ModuleList([
        #     EBTBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, 
        #              adaln_mod=decoder_adaln_mod) for _ in range(decoder_depth)])
        
        # if decoder_adaln_mod:
        #     self.decoder_norm = norm_layer(decoder_embed_dim, elementwise_affine=False, eps=1e-5)
        # else:
        #     self.decoder_norm = norm_layer(decoder_embed_dim, elementwise_affine=True, eps=1e-5)
            
        self.diffusion_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, encoder_embed_dim), requires_grad=False)

        self.t_embedder = TimestepEmbedder(encoder_embed_dim)
        # self.gt_latents_embed = nn.Linear(self.token_embed_dim, decoder_embed_dim, bias=True)

        # self.word_embedding = None
        # self.word_embedding = nn.Parameter(torch.zeros(self.cookbook_size, self.vae_embed_dim))

        self.final_layer = FinalLayer(encoder_embed_dim, final_embed_dim, self.z_proj_dim, patch_size, adaln_mod=final_layer_adaln_mod, mlp_ratio=mlp_ratio)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1) # turn off for cosim

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.ddpmloss = DDPMLoss(
            target_channels=self.vae_embed_dim,
            z_channels=decoder_embed_dim,
            # width=ddpmloss_w,
            # depth=ddpmloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self.celoss_scale = celoss_scale
        self.ddpmloss_scale = ddpmloss_scale
        self.reloss_scale = reloss_scale

        self.alpha = alpha
        self.beta = beta

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            # Linear layers
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    torch.nn.init.xavier_uniform_(module.weight)
                # elif hasattr(module, 'weight_g'):
                #     nn.init.normal_(module.weight_g, std=0.02)
                #     # module.weight_g.requires_grad = False
                #     nn.init.normal_(module.weight_v, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            # Normalization Layers
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in EBT blocks:
        self.blocks = chain(self.encoder_blocks)
        for block in self.blocks:
            if isinstance(block, EBTBlock):
                if block.adaln_mod:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0.0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0.0)

        # Initialize parameters
        # nn.init.normal_(self.z_proj.weight, std=0.02)
        # nn.init.normal_(self.encoder_embed.weight, std=0.02)
        nn.init.normal_(self.class_emb.weight, std=0.02)
        nn.init.normal_(self.fake_latent, std=0.02)
        # nn.init.normal_(self.word_embedding, std=.02)
        # nn.init.normal_(self.mask_token, std=.02)
        # nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        # nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        # nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        pos_embed = get_2d_sincos_pos_embed(self.diffusion_pos_embed.shape[-1], int(self.seq_len ** 0.5))
        self.diffusion_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out output layers:
        if self.final_layer.adaln_mod:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.normal_(self.final_layer.q_proj.weight, std=0.02)
        # nn.init.constant_(self.final_layer.q_proj.weight, 0.0)
        # nn.init.constant_(self.final_layer.q_proj.bias, 0.0)
        # nn.init.constant_(self.final_layer.logit_bias, 0.0)
        nn.init.normal_(self.final_layer.mlp[0].weight, std=0.02)
        nn.init.constant_(self.final_layer.mlp[0].bias, 0)
        nn.init.constant_(self.final_layer.mlp[-1].weight, 0)
        # nn.init.constant_(self.final_layer.mlp[-1].bias, 0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim  # will have to change if using VQGAN
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, H, W, C)
    #     """
    #     c = self.out_channels
    #     p = self.x_embedder.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #     return imgs

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz = x.shape[0]
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        # mask_rate = 1.0
        num_masked_tokens = int(np.ceil(self.seq_len * mask_rate))
        mask = torch.zeros(bsz, self.seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, self.seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, t_embedding, class_embedding):

        bsz = x.shape[0]
        
        # print(f"forward_mae_encoder - x: {x.shape}")

        # encoder projection
        # x = self.encoder_embed(x)

        # class_embeddings = class_embedding.unsqueeze(dim=1).expand(-1, self.buffer_size, -1)
        # x = torch.cat([class_embeddings, x], dim=1)

        # encoder position embedding
        x = x + self.diffusion_pos_embed
        # x = self.encoder_embed_ln(x)

        # dropping
        # x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        cond = (t_embedding + class_embedding).unsqueeze(dim=1)
        
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for block in self.encoder_blocks:
                x = checkpoint(block, x, cond, use_reentrant=False)
        else:
            for block in self.encoder_blocks:
                x = block(x, cond)

        # x = self.encoder_norm(x)

        # x = x[:, self.buffer_size:]
        # x = x + self.diffusion_pos_embed_learned

        return x
    
    # def forward_mae_decoder(self, x, mask, t_embedding, class_embedding):

    #     bsz, seq_len, embed_dim = x.shape

    #     # decoder projection
    #     x = self.decoder_embed(x)

    #     # # replace masked position with mask tokens
    #     # mask_with_buffer = torch.cat([torch.zeros(bsz, self.buffer_size, dtype=x.dtype, device=x.device), mask], dim=1)
    #     # mask_tokens = self.mask_token.expand(bsz, seq_len, -1).to(x.dtype)
    #     # x = (mask_with_buffer.unsqueeze(dim=-1) * mask_tokens)  + ((1.0 - mask_with_buffer.unsqueeze(dim=-1)) * x)

    #     # decoder position embedding
    #     # x = x + self.decoder_pos_embed_learned
    #     # # # x = self.decoder_embed_ln(x)

    #     # apply Transformer blocks
    #     cond = (t_embedding + class_embedding).unsqueeze(dim=1)

    #     if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
    #         for block in self.decoder_blocks:
    #             x = checkpoint(block, x, cond, use_reentrant=False)
    #     else:
    #         for block in self.decoder_blocks:
    #             x = block(x, cond)

    #     x = self.decoder_norm(x)

    #     # x = x[:, self.buffer_size:]
    #     # x = x + self.diffusion_pos_embed_learned
        
    #     return x

    def forward(self, imgs, labels, gt_indices, cookbook, warmup):

        # print(f"MAR forward - imgs: {imgs.shape}")

        # patchify and mask (drop) tokens
        # x = self.patchify(imgs)  # [B, C, H, W] -> [B, L, C]

        x = imgs  # [B, C, H, W]
        bsz, c, h, w = x.shape
        orders = self.sample_orders(bsz=imgs.shape[0])
        mask = self.random_masking(x, orders)

        # class embed
        class_embedding = self.class_emb(labels)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz, device=x.device) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        # compute diffusion loss
        ddpmloss, celoss, reloss, logitsnorm, qnorm, pi, scorenorm, tembnorm, scale = self.ddpmloss(self, x, mask, class_embedding, cookbook, gt_indices, warmup)

        if mask is not None:
            # if mask.dim() > 1:
            #     mask = mask.flatten(start_dim=0, end_dim=1)
            mask_spatial = mask.view(bsz, self.seq_h, self.seq_w).repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
            mask = mask_spatial.reshape(bsz, -1)
            ddpmloss_masked = (ddpmloss * mask).sum() / (mask.sum() + 1e-8)
            celoss_masked = (celoss * mask).sum() / (mask.sum() + 1e-8)
            reloss_masked = (reloss * mask).sum() / (mask.sum() + 1e-8)

        logitsnorm_mean = logitsnorm.mean()
        qnorm_mean = qnorm.mean()
        pimax_mean = pi.max(dim=-1)[0].mean()
        scorenorm_mean = scorenorm.mean()
        tembnorm_mean = tembnorm.mean()
        scale_max = scale.max()

        loss_mean = self.ddpmloss_scale * ddpmloss_masked + self.celoss_scale * celoss_masked + self.reloss_scale * reloss_masked

        return loss_mean, ddpmloss_masked, celoss_masked, reloss_masked, logitsnorm_mean, qnorm_mean, pimax_mean, scorenorm_mean, tembnorm_mean, scale_max

    def sample_tokens(self, eval_bsz, cookbook, num_iter=64, cfg=1.0, cfg_schedule="linear", temperature=1.0, imgs=None, labels=None, gt_indices=None, sampling_mode="diffusion", progress=False):

        # init and sample generation orders
        mask = torch.ones(eval_bsz, self.seq_len, dtype=torch.bool).cuda()
        tokens = torch.zeros(eval_bsz, self.vae_embed_dim, self.token_h, self.token_w).cuda()
        orders = self.sample_orders(eval_bsz)

        # if imgs is not None:
        #     print(f"Sample Tokens - tokens: {imgs.shape}")
        #     imgs = self.patchify(imgs)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                # print(f"Sample Tokens - labels: {labels.shape}")
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(eval_bsz, 1)

            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(eval_bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # print(f"Sample Tokens - tokens: {tokens.shape}, class_embedding: {class_embedding.shape}, mask: {mask.shape}")

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # print(f"Sample Tokens - mask_ratio: {mask_ratio}, mask_len: {mask_len}")

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
            
            # print(f"Sample Tokens - mask_len: {mask_len}")

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, eval_bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:eval_bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:eval_bsz].bool(), mask_next.bool())
            # mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            # z = z[mask_to_pred.nonzero(as_tuple=True)]

            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.ddpmloss.sample(self, tokens, mask, mask_to_pred, class_embedding, cookbook, temperature, cfg_iter, mode=sampling_mode, imgs=imgs)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            # print(f"Sample Tokens - cur_tokens: {cur_tokens.shape}, sampled_token_latent: {sampled_token_latent.shape}, mask_to_pred: {mask_to_pred.shape}")

            # cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]
            # mask_to_pred_spatial = mask_to_pred.to(cur_tokens.dtype).view(eval_bsz, self.token_h, self.token_w)
            mask_to_pred_spatial = mask_to_pred.to(cur_tokens.dtype).view(eval_bsz, self.seq_h, self.seq_w).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
            cur_tokens = (1.0 - mask_to_pred_spatial.unsqueeze(dim=1)) * cur_tokens + mask_to_pred_spatial.unsqueeze(dim=1) * sampled_token_latent

            tokens = cur_tokens.clone()
            mask = mask_next

            # print(f"Sample Tokens - num_masked: {mask.nonzero().size(0)}, num_unmasked: {(~mask).nonzero().size(0)}, num_mask_to_pred: {mask_to_pred.nonzero().size(0)}, sampling_mode: {sampling_mode}")

        # print(f"Sample Tokens - tokens: {tokens.shape}")

        # unpatchify
        # tokens = self.unpatchify(tokens)
        return tokens

class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, in_channels, model_channels, out_channels, patch_size, adaln_mod=False, min_logit_scale=0.0, max_logit_scale=1.0, mlp_ratio=4):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.adaln_mod = adaln_mod
        print(f"FinalLayer - adaln_mod: {self.adaln_mod}")

        self.norm_in = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-5)

        if self.adaln_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_channels, in_channels * 2, bias=True)
            )

        # self.norm_out = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-5)
        self.q_proj = nn.Linear(in_channels, model_channels, bias=False)

        self.mlp = nn.Sequential(
            nn.LayerNorm(out_channels, elementwise_affine=True, eps=1e-5),
            nn.Linear(out_channels, out_channels * mlp_ratio, bias=True),
            nn.SiLU(),
            nn.Linear(out_channels * mlp_ratio, out_channels, bias=False)
        )

        # self.logit_scale = nn.Parameter(torch.tensor(1.0))
        # self.min_logit_scale = min_logit_scale
        # self.max_logit_scale = max_logit_scale

        # self.logit_bias = nn.Parameter(torch.zeros(1, 1, cookbook_size))

    def forward(self, mar, x, t_embedding, class_embedding, cookbook_embedding=None):

        bsz, l, d = x.shape

        x = self.norm_in(x)

        if self.adaln_mod:
            cond = (t_embedding + class_embedding).unsqueeze(dim=1)
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
            x = modulate(x, shift, scale)

        q = self.q_proj(x)
        # q_gelu = self.gelu(q_proj)

        # unpatchify [B,L,D] -> [B, L*P^2, D/P^2]
        q_upsampled = self.unpatchify(q)

        word_embedding = cookbook_embedding + self.mlp(cookbook_embedding)
        word_embedding = F.normalize(word_embedding, dim=-1)

        # q_norm = F.normalize(q_upsampled, p=2.0, dim=-1)
        # c_norm = self.rms(cookbook)

        # print(f"FinalLayer - q_upsampled: {q_upsampled.shape}, cookbook_embedding: {cookbook_embedding.shape}")

        if mar.beta == 0:
            logits = torch.zeros(bsz, q_upsampled.shape[1], mar.cookbook_size, dtype=x.dtype, device=x.device, requires_grad=False)
        else:
            logits = torch.einsum('B L D, K D -> B L K', q_upsampled, word_embedding)

        pi = torch.softmax(logits, dim=-1)
        # v = torch.einsum('B L K, K D -> B L D', pi, cookbook_embedding)

        return logits, q_upsampled, pi

    def unpatchify(self, x):
        bsz, l, d = x.shape
        h_ = w_ = int(l ** 0.5)
        p = self.patch_size
        c = d // p**2

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nhpwqc', x)
        x = x.reshape(bsz, h_ * p * w_ * p, c)
        return x  # [n, h * w, c]

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # print(torch.nn.functional.softmax(logits, dim=-1).detach().mean())
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(-1))
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

class EBTBlock(nn.Module):
    """
    A EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = partial(nn.GELU, approximate='tanh'),
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            adaln_mod: bool = False,
            cross_attn: bool = False,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()

        self.cross_attn = cross_attn
        # self.layer_scale1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # if cross_attn:
        #     self.attn = HopfieldAttention(
        #         dim,
        #         num_heads=num_heads,
        #         qkv_bias=qkv_bias,
        #         qk_norm=qk_norm,
        #         scale_norm=scale_attn_norm,
        #         proj_bias=proj_bias,
        #         attn_drop=attn_drop,
        #         proj_drop=proj_drop,
        #         norm_layer=norm_layer,
        #         hetero=True,
        #     )
        #     self.normc = norm_layer(dim, elementwise_affine=False, eps=1e-5)
        #     self.num_adaLN_params = 8
        # else:

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        # self.layer_scale2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.adaln_mod = adaln_mod
        print(f"EBTBlock - adaln_mod: {self.adaln_mod}")

        if self.adaln_mod: 
            self.norm1 = norm_layer(dim, elementwise_affine=False, eps=1e-5)
            self.norm2 = norm_layer(dim, elementwise_affine=False, eps=1e-5)

            self.num_adaLN_params = 6
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, self.num_adaLN_params * dim, bias=True)
            )
        else:
            self.norm1 = norm_layer(dim, elementwise_affine=True, eps=1e-5)
            self.norm2 = norm_layer(dim, elementwise_affine=True, eps=1e-5)

    def forward(self, x, y, c=None):

        # if self.cross_attn:
        #     shift_q, scale_q, shift_k, scale_k, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(self.num_adaLN_params, dim=-1)
        #     q = modulate(self.norm1(q), shift_q, scale_q)
        #     k = modulate(self.normc(k), shift_k, scale_k)
        #     with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        #         energy, grad_q, grad_k = self.attn(q, k) # needed to set this as regular sdpa from pt didnt support higher order gradients
        #     # energy = 0.5 * q.pow(2).sum(dim=-1, keepdim=True) + (gate_attn * energy)
        #     q = q + gate_attn * self.drop_path1(self.layer_scale1(grad_q))
        #     q = q + gate_mlp * self.drop_path2(self.layer_scale2(self.mlp(modulate(self.norm2(q), shift_mlp, scale_mlp))))
        # else:

        if self.adaln_mod:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(self.num_adaLN_params, dim=-1)
            x_norm = self.norm1(x)
            x_mod = modulate(x_norm, shift_attn, scale_attn)
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            with sdpa_kernel(SDPBackend.MATH):
                attn = self.attn(x_mod)
            x = x + gate_attn * attn
            x_norm = self.norm2(x)
            x_mod = modulate(x_norm, shift_mlp, scale_mlp)
            x = x + gate_mlp * self.mlp(x_mod)
        else:
            x_norm = self.norm1(x)
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            with sdpa_kernel(SDPBackend.MATH):
                attn = self.attn(x_norm)
            x = x + attn
            x_norm = self.norm2(x)
            x = x + self.mlp(x_norm)
        return x

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_freq

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def mar_tiny(**kwargs):
    model = MAR(
        encoder_embed_dim=384, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=4,
        final_embed_dim=384,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
    return model


def mar_small(**kwargs):
    model = MAR(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        final_embed_dim=512,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
    return model


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        final_embed_dim=768,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        final_embed_dim=1024,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1152, encoder_depth=28, encoder_num_heads=16,
        decoder_embed_dim=1152, decoder_depth=28, decoder_num_heads=16,
        final_embed_dim=1152,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
    return model

# def mar_large(**kwargs):
#     model = MAR(
#         encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
#         decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
#         final_embed_dim=1024,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
#     return model


# def mar_huge(**kwargs):
#     model = MAR(
#         encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
#         decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
#         final_embed_dim=1280,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-5), **kwargs)
#     return model
