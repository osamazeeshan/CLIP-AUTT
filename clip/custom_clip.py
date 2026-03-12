
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
import seaborn as sns
import os

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.biovid_prompts import biosub_classes
from data.bah_prompts import bahssub_classes
from .transformer import TemporalTransformer
import copy
import types

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        """
        Compute Grad-CAM for given input and target class.
        """
        logits = self.model(input_tensor)[0]  # forward
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop for selected class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # global average pool
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def patch_clip_attention(block):
    """
    Monkey-patch a CLIP ResidualAttentionBlock.attn.forward
    so it returns (attn_out, attn_probs).
    """
    attn = block.attn

    if getattr(attn, "_is_patched", False):
        return  # already patched

    def forward_with_probs(self, query, key, value, need_weights=True, attn_mask=None):
        # identical to torch.nn.functional.multi_head_attention_forward
        # but we capture the weights
        import torch.nn.functional as F
        attn_out, attn_probs = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            need_weights=True,
            attn_mask=attn_mask,
            use_separate_proj_weight=False
        )
        return attn_out, attn_probs

    # replace the forward method
    attn.forward = types.MethodType(forward_with_probs, attn)
    attn._is_patched = True


class VClip(nn.Module):
    def __init__(
            self,
            arch,
            device,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 4,
            dim_forward: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_forward = dim_forward
        # model, _ = load("ViT-B/32", device=device, jit=False)
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        for name, param in clip.named_parameters():
            param.requires_grad = False
        self.backbone = clip
        self.temporal = TemporalTransformer(
            input_dim=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_dim=d_model,
            dim_head=dim_forward
        )
        self.logit_scale = nn.Parameter(self.backbone.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True
    
    def forward(self, x, text, device):
        image_features = self.encode_video(x)

        # text_features = self.encode_text(text)
        text_features = self.encode_text(tokenize(text).to(device))

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
    def encode_video(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        v = self.backbone.encode_image(x).reshape(B, T, -1)
        v = self.temporal(v)
        v = v[:, 0]
        return v
    def encode_text(self, text):
        encoded_text = self.backbone.encode_text(text)
        return encoded_text



class TemporalGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()  # shape [B, T, D]
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, inputs, texts, target_class=None):
        logits, _ = self.model(inputs, texts)
        if target_class is None:
            target_class = logits.argmax(dim=-1)

        self.model.zero_grad()
        logits[:, target_class].sum().backward(retain_graph=True)

        grads = self.gradients.mean(dim=1)  # temporal average over time steps
        cams = (self.activations * grads.unsqueeze(1)).sum(dim=-1)
        cams = F.relu(cams)
        cams = (cams - cams.min()) / (cams.max() + 1e-8)
        return cams.cpu().numpy()  # shape [B, T]


'''
CLIP IMAGE AND VIDEO BASED MODELS

'''
class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # seq_len = prompts.size(1)
        # pos_embed = self.positional_embedding[:seq_len,:].unsqueeze(0)
        # x = prompts + pos_embed
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]

            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class AUPromptLearner(nn.Module):
    """
    Learnable prompt encoder for Action Units (AUs).
    Each AU has its own trainable prompt context.
    """
    def __init__(self, clip_model, au_list, n_ctx=8, ctx_init=None):
        super().__init__()

        self.device = clip_model.visual.conv1.weight.device
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_aus = len(au_list)
        self.n_ctx = n_ctx
        self.au_list = au_list

        # 🔹 Initialize context tokens (trainable)
        if ctx_init:
            print(f"[INFO] Initializing AU context with phrase: '{ctx_init}'")
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            print(f"[INFO] Random init of {n_ctx} context tokens per AU.")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        # one learnable context per AU (duplicate the same initialization)
        ctx_vectors = ctx_vectors.unsqueeze(0).repeat(self.n_aus, 1, 1)
        self.ctx = nn.Parameter(ctx_vectors)  # [num_aus, n_ctx, D]

        # keep a copy for reset
        self.ctx_init_state = ctx_vectors.detach().clone()

        # 🔹 Build AU prompt templates and tokenize
        au_prompts = [au.replace("_", " ") + "." for au in au_list]
        tokenized = torch.cat([tokenize(p) for p in au_prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)

        # first token [SOS], last token [EOS]
        self.register_buffer("prefix", embedding[:, :1, :])
        self.register_buffer("suffix", embedding[:, -1:, :])

        self.tokenized_prompts = tokenized
        print(f"[INFO] AUPromptLearner initialized with {self.n_aus} prompts × {self.n_ctx} context tokens.")

    def reset(self):
        """Reset learnable context tokens to their initial state."""
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)
        print("[INFO] AU prompt contexts reset to initial state.")

    def reset_prompts(self, new_au_list, clip_model):
        """Optional: reinitialize prompt templates for a new AU list."""
        self.au_list = new_au_list
        au_prompts = [au.replace("_", " ") + "." for au in new_au_list]
        tokenized = torch.cat([tokenize(p) for p in au_prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)
        self.prefix = embedding[:, :1, :]
        self.suffix = embedding[:, -1:, :]
        self.tokenized_prompts = tokenized
        print(f"[INFO] AU prompt templates reset for {len(new_au_list)} AUs.")

    def forward(self):
        """Construct trainable AU prompts and pad to 77 tokens for CLIP compatibility."""
        prefix = self.prefix   # [num_aus, 1, D]
        suffix = self.suffix   # [num_aus, 1, D]
        ctx = self.ctx         # [num_aus, n_ctx, D]

        # Construct AU prompt sequence: [SOS] + ctx + [EOS]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # [num_aus, n_ctx+2, D]

        # ✅ Pad to 77 tokens (CLIP text encoder expected length)
        max_len = 77
        seq_len = prompts.size(1)
        if seq_len < max_len:
            pad_len = max_len - seq_len
            pad = torch.zeros(
                prompts.size(0), pad_len, prompts.size(2),
                device=prompts.device, dtype=prompts.dtype
            )
            prompts = torch.cat([prompts, pad], dim=1)  # [num_aus, 77, D]

        return prompts


class ClipTestTimeTuning(nn.Module):

    def __init__(self, device, classnames, batch_size,
                 criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                 num_aus=30, num_classes=2, text_hidden=512, clf_hidden=256):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.clip = clip
        self.device = device

        # 🔹 text adapter for AU prompts (trainable)
        d = self.image_encoder.output_dim  # CLIP embed dim
        self.text_adapter = nn.Sequential(
            nn.Linear(d, text_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden, d)
        )

        # 🔹 AU classifier head (trainable)
        self.au_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # 🔹 Subject-specific adapter (very small; train per subject)
        # for example: a learnable affine transform on AU features
        self.subject_adapter = nn.Sequential(
            nn.Linear(num_classes, num_classes)
        )

        # fusion weight for simple weighting of the two logits
        self.fusion_alpha = 0.5  # can be tuned or learned
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    @torch.no_grad()
    def get_text_features_from_prompts(self, prompt_list):
        """
        Encode arbitrary custom prompt strings into normalized text features.
        prompt_list: list of strings
        returns: tensor [len(prompt_list), D]
        """
        tokenized_prompts = torch.cat([tokenize(p) for p in prompt_list]).to(self.image_encoder.conv1.weight.device)

        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized_prompts).type(self.text_encoder.dtype)
        t_features = self.text_encoder(embedding, tokenized_prompts)
        return t_features / t_features.norm(dim=-1, keepdim=True)  # [N,D]
        
    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    # AU pathway logits (text adapter + AU classifier)
    def au_pathway_logits(self, image, au_prompts):
        with torch.no_grad():
            img_features = self.image_encoder(image.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_features = F.normalize(img_features, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        sim = img_features @ base_text_embeds.T  # [B, num_aus]
        logits = self.au_classifier(sim)       # [B, num_classes]
        return logits

    # fused logits with subject-specific adapter on top
    def fused_logits(self, image, au_prompts, alpha=None):
        if alpha is None:
            alpha = self.fusion_alpha
        logits_clip = self.inference(image)             # [B,C]
        logits_au = self.au_pathway_logits(image, au_prompts)  # [B,C]
        fused = alpha * logits_au + (1 - alpha) * logits_clip  # [B,C]
        # subject-specific adapter (train per subject)
        fused = self.subject_adapter(fused)  # [B,C]
        return fused

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input, au_prompts=None, mode="clip"):
        """
        mode:
          "clip"  -> original CLIP logits
          "au"    -> AU pathway logits
          "fused" -> weighted combination of the two + subject adapter
        """
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            if mode == "clip":
                return self.inference(input)
            elif mode == "au":
                assert au_prompts is not None
                return self.au_pathway_logits(input, au_prompts)
            elif mode == "adapt":
                assert au_prompts is not None
                return self.fused_logits(input, au_prompts)

    def compute_au_similarities(self, images, au_prompts, device):
        """
        Compute AU similarity vectors for a batch of images.
        Returns: [B,num_aus] tensor
        """
        with torch.no_grad():
            img_features = self.image_encoder(images.type(self.dtype))
            img_features = F.normalize(img_features, dim=-1)

            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted_embeds = self.text_adapter(base_text_embeds)  # keep grad if training adapter
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)

        sim = img_features @ adapted_embeds.T  # [B,num_aus]
        return sim
        
    def get_au_similarity(self, image_features, au_prompts):
        """
        image_features: [B,D] CLIP image embeddings (normalized)
        au_prompts: list of AU prompt strings
        returns: [B, num_aus] similarity scores
        """
        with torch.no_grad():
            base_text_embeds = self.clip.encode_text(
                tokenize(au_prompts).to(self.device)
            )  # [num_aus,D]
        adapted_embeds = self.text_adapter(base_text_embeds)      # [num_aus,D]
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)
        sim = image_features @ adapted_embeds.T                  # [B,num_aus]
        return sim
    

class ClipTestTimeVideoTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, au_prompts,
                 criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                 num_aus=46, num_classes=2, text_hidden=512, clf_hidden=256,
                 d_model=192, num_layers=2, nhead=4, dim_forward=512,
                 delta_stride=1):
        super(ClipTestTimeVideoTuning, self).__init__()

        # -------------------------------------------------------
        # 1️⃣ Load CLIP backbone
        # -------------------------------------------------------
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        # self.logit_scale = clip.logit_scale.data
        self.prompt_learner = PromptLearner(
            clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls
        )
        self.au_prompt_learner = AUPromptLearner(clip, au_prompts, n_ctx) 
        self.device = device
        self.clip = clip
        self.classnames = classnames

        self.logit_scale = nn.Parameter(clip.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True

        # -------------------------------------------------------
        # 2️⃣ Trainable AU adapter + classifier
        # -------------------------------------------------------
        d = self.image_encoder.output_dim  # CLIP embed dim

        # Text adapter (trainable)
        self.text_adapter = nn.Sequential(
            nn.Linear(d, text_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden, d)
        )

        # AU classifier (trainable)
        self.au_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # 🔹 Subject-specific adapter (very small; train per subject)
        # for example: a learnable affine transform on AU features
        self.subject_adapter = nn.Sequential(
            nn.Linear(num_classes, num_classes)
        )

        # fusion weight for simple weighting of the two logits
        self.fusion_alpha = 0.5  # can be tuned or learned

        # temporal_classifier (trainable)
        self.temporal_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden), # IMPORANT CHNAGE BACK TO num_aus
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # -------------------------------------------------------
        # 3️⃣ Temporal Transformer for AU sequence modeling
        # -------------------------------------------------------
        self.temporal_proj = nn.Linear(d*2, d_model)
        self.temporal = nn.Conv1d(d, d, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.1)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d) * 0.02)
        # how frequently to compute ΔA_t (e.g., every 1, 2, or 3 frames)
        self.delta_stride = delta_stride
        self.class_to_au = nn.Linear(num_classes, num_aus, bias=False)

    def visualize_video_text_au_tsne_paper(self, video, au_prompts, adapt_tar, key_frame_sel, key_frames, labels, sub_id="00", device="cuda"):
        """
        Visualize t-SNE embeddings of video features, AU text embeddings (before), 
        and AU adapter embeddings (after).
        """

        self.eval()
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            # --- AU text embeddings ---
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

            adapted_text_embeds = self.text_adapter(base_text_embeds)
            adapted_text_embeds = F.normalize(adapted_text_embeds, dim=-1)

            # --- Visual (temporal) embeddings ---
            img_feats = self.image_encoder(x.type(self.dtype))
            img_feats = img_feats.reshape(B, T, -1)

            selected_feats = []
            if adapt_tar and key_frame_sel:
                for b in range(B):
                    frame_feats = img_feats[b]
                    key_idx = self.select_key_consec_frames(frame_feats, adapted_text_embeds, top_k=key_frames)
                    selected_feats.append(frame_feats[key_idx])
                max_T = max(len(f) for f in selected_feats)
                selected_feats = torch.stack([
                    F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
                ])
            else:
                selected_feats = img_feats

            v = self.temporal(selected_feats.transpose(1, 2))
            v = F.gelu(v)
            visual_embeds = v.mean(dim=-1)
            visual_embeds = F.normalize(visual_embeds, dim=-1)

        # Cosine similarities: 32×46 matrices
        sim_before = F.cosine_similarity(
            visual_embeds.unsqueeze(1), base_text_embeds.unsqueeze(0), dim=-1
        )
        sim_after = F.cosine_similarity(
            visual_embeds.unsqueeze(1), adapted_text_embeds.unsqueeze(0), dim=-1
        ) 

        # ---- Convert to numpy ----
        sim_before_np = sim_before.cpu().numpy()
        sim_after_np = sim_after.cpu().numpy()
        label_vals = labels.detach().cpu().numpy()

        # ---- Run t-SNE separately for before and after ----
        tsne_before = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")
        tsne_after = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")

        X_before = tsne_before.fit_transform(sim_before_np)
        X_after = tsne_after.fit_transform(sim_after_np)

        # ---- Create side-by-side CVPR-style figure ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # --- Before Adapter ---
        scatter1 = axes[0].scatter(
            X_before[:, 0], X_before[:, 1],
            c=label_vals, cmap="coolwarm", s=60, edgecolor="k", alpha=0.85
        )
        axes[0].set_title("Before AU Adapter", fontsize=18, weight="semibold")

        # --- After Adapter ---
        scatter2 = axes[1].scatter(
            X_after[:, 0], X_after[:, 1],
            c=label_vals, cmap="coolwarm", s=60, edgecolor="k", alpha=0.85
        )
        axes[1].set_title("After AU Adapter", fontsize=18, weight="semibold")

        # --- Hide ticks and tick labels, keep box frames ---
        for ax in axes:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color("black")

        plt.subplots_adjust(wspace=0.25)
        plt.suptitle(
            "t-SNE of Video–AU Similarity Patterns\nBefore and After AU Adapter",
            fontsize=20, weight="bold"
        )

        save_path = "visual/tsne/tsne_dual_video_similarity_"+str(sub_id)+".png"
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Dual t-SNE figure saved to: {save_path}")

        plt.show()


    # ===========================================================
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_states(self):
        """
        Reset runtime states (hidden, cache, buffer) 
        for temporal modules after each video.
        Keeps pretrained weights intact.
        """
        with torch.no_grad():
            modules = [self.text_adapter, self.temporal, self.temporal_classifier]
            for module in modules:
                module.zero_grad(set_to_none=True)
    
    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    # ===========================================================
    @torch.no_grad()
    def get_text_features_from_prompts(self, prompt_list):
        """Compute text embeddings from arbitrary prompt strings."""
        tokenized = torch.cat([tokenize(p) for p in prompt_list]).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized).type(self.text_encoder.dtype)
        t_features = self.text_encoder(embedding, tokenized)
        return F.normalize(t_features, dim=-1)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def get_au_features(self):
        prompts = self.au_prompt_learner()
        tokenized_prompts = self.au_prompt_learner.tokenized_prompts
        au_features = self.text_encoder(prompts, tokenized_prompts)
        au_features_norm = F.normalize(au_features, dim=-1)

        return au_features_norm
    # ===========================================================
    def compute_au_similarities(self, video, au_prompts):
        """
        Compute per-frame AU similarity vectors for a video.
        video: [B, T, 3, H, W]
        au_prompts: list of AU text prompts
        """
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            img_feats = self.image_encoder(flat.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_feats = F.normalize(img_feats, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted = self.text_adapter(base_text_embeds)
        adapted = F.normalize(adapted, dim=-1)

        sim = img_feats @ adapted.T
        return sim.view(B, T, -1)  # [B, T, num_aus]

    # ===========================================================
    def forward_temporal(self, video, au_prompts):
        """
        Perform forward pass for temporal FER.
        During training, this is where gradients flow.
        """
        B, T, _, _, _ = video.shape

        A = self.compute_au_similarities(video, au_prompts)  # [B, T, num_aus]

        if self.delta_stride > 1:
            dA = A[:, self.delta_stride:, :] - A[:, :-self.delta_stride, :]
            pad = torch.zeros_like(A[:, :self.delta_stride, :])
            dA = torch.cat([pad, dA], dim=1)
        else:
            dA = torch.diff(A, dim=1, prepend=A[:, 0:1, :])

        X = torch.cat([A, dA], dim=-1)      # [B, T, 2*num_aus]
        X = self.temporal_proj(X)           # [B, T, d_model]

        out = self.temporal(X)              # [B, T+1, d_model]
        z_cls = self.temporal_norm(out[:, 0, :])  # [B, d_model] (CLS token)

        logits = self.temporal_classifier(z_cls)
        return logits

    def select_key_frames(self, video_feats, adapted_embeds, top_k=5):
        """
        Select key frames using entropy-based confidence on AU similarities.
        
        Args:
            video_feats: [T, D] visual features (CLIP frame embeddings)
            adapted_embeds: [num_AUs, D] normalized AU text embeddings
            top_k: number of frames to keep

        Returns:
            List[int]: indices of the most confident (low-entropy) frames
        """
        T = video_feats.size(0)
        if T <= top_k:
            return list(range(T))

        # --- Compute AU similarity per frame ---
        video_feats_norm = F.normalize(video_feats, dim=-1)
        au_sim = video_feats_norm @ adapted_embeds.T  # [T, num_AUs]

        # --- Classifier logits per frame ---
        logits_per_frame = self.temporal_classifier(au_sim)
        probs = F.softmax(logits_per_frame, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # --- Select lowest entropy frames (highest confidence) ---
        top_idx = torch.topk(-entropy, k=top_k).indices
        return sorted(top_idx.cpu().tolist())

    def select_key_consec_frames(self, video_feats, adapted_embeds, top_k=5, percentile=0.3):
        """
        Select top-K consecutive frames with lowest average entropy.
        Args:
            video_feats: [T, D] CLIP visual frame features (normalized)
            adapted_embeds: [num_AUs, D] AU text embeddings (after adapter + norm)
            top_k: number of consecutive frames to keep
        Returns:
            List[int]: indices of selected consecutive frames
        """
        T = video_feats.size(0)
        if T <= top_k:
            return list(range(T))

        # --- Compute per-frame AU similarity and logits ---
        au_sim = F.normalize(video_feats, dim=-1) @ adapted_embeds.T
        logits_per_frame = self.temporal_classifier(au_sim)

        # --- Compute entropy for each frame ---
        probs = F.softmax(logits_per_frame, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) 

        # --- Sliding window average entropy ---
        window_entropies = torch.stack([
            entropy[i:i+top_k].mean() for i in range(T - top_k + 1)
        ])  # [T - top_k + 1]

        # --- Find window with lowest mean entropy (most confident segment) ---
        start_idx = torch.argmin(window_entropies).item()
        selected_idx = list(range(start_idx, start_idx + top_k))

        return selected_idx


    def select_key_consec_frames_temporal(self, video_feats, adapted_embeds, top_k=16):
        """
        Select top-K consecutive frames with lowest entropy calculated after temporal aggregation.
        Args:
            video_feats: [T, D] CLIP visual frame features (normalized)
            adapted_embeds: [num_AUs, D] AU text embeddings (after adapter + norm)
            top_k: number of consecutive frames to keep
        Returns:
            List[int]: indices of selected consecutive frames
        """
        T = video_feats.size(0)
        
        # If video is shorter than window size, return all frames
        if T <= top_k:
            return list(range(T))

        window_entropies = []

        # Iterate through all possible consecutive windows
        for i in range(T - top_k + 1):
            # 1. Select window features: [top_k, D]
            window_feats = video_feats[i : i + top_k]
            
            # 2. Add batch dimension: [1, top_k, D]
            # The temporal module snippet expects [B, D, T] input (transposed)
            window_feats_batch = window_feats.unsqueeze(0) 
            
            # 3. Apply Temporal Module
            # Input transposes to [1, D, top_k]
            v = self.temporal(window_feats_batch.transpose(1, 2))
            v = F.gelu(v)
            
            # 4. Aggregate temporally: [1, D]
            visual_embeds = v.mean(dim=-1)
            
            # Normalize the aggregated visual embedding
            visual_embeds = F.normalize(visual_embeds, dim=-1)
            
            # 5. Compute AU Similarity: [1, num_AUs]
            # adapted_embeds is [num_AUs, D], so we transpose it
            au_sim = visual_embeds @ adapted_embeds.T
            
            # 6. Apply Temporal Classifier to get logits: [1, num_classes]
            logits = self.temporal_classifier(au_sim)
            
            # 7. Compute Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) # Scalar [1]
            
            window_entropies.append(entropy.item())

        # --- Find window with lowest entropy (most confident aggregated segment) ---
        start_idx = torch.argmin(torch.tensor(window_entropies)).item()
        selected_idx = list(range(start_idx, start_idx + top_k))

        return selected_idx


    def temporal_clip_au_forward(self, video, au_prompts, class_prompts, adapt_tar=False, key_frame_sel=False, train_whole_clip=False, key_frames=16):
        """
        Simple temporal CLIP forward:
        1. Extract CLIP embeddings per frame.
        2. Model temporal structure via transformer.
        3. Compute similarity with AU prompts (or class prompts).

        Args:
            video: [B, T, 3, H, W] tensor
            au_prompts: list of AU text prompts (len = num_aus)

        Returns:
            logits_per_video: [B, num_aus]
        """
        B, T, C, H, W = video.shape

        # 1️⃣ Encode AU text prompts
        if train_whole_clip:
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))
        else:
            with torch.no_grad():
                base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))  # [num_aus, D]
            if class_prompts is not None:
                if adapt_tar:
                    base_cls_text_embeds = self.get_text_features()
                else:
                    base_cls_text_embeds = self.clip.encode_text(tokenize(class_prompts).to(self.device))  # [num_aus, D]
            else:
                base_cls_text_embeds = None

        # ---- AU Prompt Tuning
        if adapt_tar:
            base_text_embeds = self.get_au_features()
        text_embeds = self.text_adapter(base_text_embeds) 
        text_embeds_norm = F.normalize(text_embeds, dim=-1)

        x = video.reshape(B * T, C, H, W)
        if train_whole_clip:
            img_feats = self.image_encoder(x.type(self.dtype))
        else:
            with torch.no_grad():
                img_feats = self.image_encoder(x.type(self.dtype))
        img_feats = img_feats.reshape(B, T, -1)

        selected_feats = []
        if adapt_tar and key_frame_sel:
            for b in range(B):
                frame_feats = img_feats[b]  # [T, 512]
                # key_idx = self.select_key_consec_frames(frame_feats, text_embeds_norm, top_k=key_frames) # Biovid=16
                '''
                    [INFO] Selecting Window using Temporal Module: window->AU-sim->EmoClassifier->Entropy: select frame window with lowest entropy										
                '''
                key_idx = self.select_key_consec_frames_temporal(frame_feats, text_embeds_norm, top_k=key_frames) 
                selected_feats.append(frame_feats[key_idx])

            max_T = max(len(f) for f in selected_feats)
            selected_feats = torch.stack([
                F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
            ]) 

        v = self.temporal(selected_feats.transpose(1,2) if key_frame_sel else img_feats.transpose(1,2))
        v = F.gelu(v)
        visual_embeds = v.mean(dim=-1)
        visual_embeds_norm = F.normalize(visual_embeds, dim=-1)

        temp = 5.00
        au_sim = temp * (visual_embeds_norm @ text_embeds_norm.T)
        if adapt_tar:
            au_sim.retain_grad()

        if base_cls_text_embeds is not None:
            base_cls_text_embeds = F.normalize(base_cls_text_embeds, dim=-1)
            cls_sim = temp * (visual_embeds_norm @ base_cls_text_embeds.T)
            alpha=0.5

        logits = self.temporal_classifier(au_sim)   # [B, num_classes]
        if base_cls_text_embeds is not None:
            cls_temp = 0.3
            logits = ((1-cls_temp)*logits) + (cls_temp * cls_sim)
        return logits, au_sim, visual_embeds_norm
                
    def temporal_clip_au_testime(self, video, au_prompts, class_prompts, key_frame_sel=False, key_frames=16):
        # 1 Get AU logits and Temporal Video Embeddings using 1DCNN for temporal
        logits_au, au_sim, visual_embeds_norm = self.temporal_clip_au_forward(video, au_prompts, class_prompts, adapt_tar=True, key_frame_sel=key_frame_sel, key_frames=key_frames)
        return logits_au, au_sim

    # ===========================================================
    def forward(self, x, au_prompts=None, class_prompts=None, mode="temporal", adapt_target=False, key_frame_sel=False, key_frames=16, train_whole_clip=False):
        """
        mode:
          - 'temporal': sequence modeling for video (train AU adapter + AU classifier + transformer)
          - 'clip': standard CLIP image-based path
        """
        if mode == "temporal":
            if adapt_target:
                return self.temporal_clip_au_testime(x, au_prompts, class_prompts, key_frame_sel=key_frame_sel, key_frames=key_frames)
            logits, au_sim, _ = self.temporal_clip_au_forward(x, au_prompts, class_prompts, train_whole_clip=train_whole_clip)
            return logits, au_sim 

        elif mode == "clip":
            with torch.no_grad():
                img_features = self.image_encoder(x.type(self.dtype))
            text_features = self.get_text_features_from_prompts(au_prompts)
            img_features = F.normalize(img_features, dim=-1)
            logits = self.logit_scale.exp() * img_features @ text_features.t()
            return logits

     
    # AU pathway logits (text adapter + AU classifier)
    def au_pathway_logits(self, image, au_prompts):
        with torch.no_grad():
            img_features = self.image_encoder(image.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_features = F.normalize(img_features, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted_embeds = self.text_adapter(base_text_embeds)  # trainable
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)

        sim = img_features @ adapted_embeds.T  # [B, num_aus]
        logits = self.au_classifier(sim)       # [B, num_classes]
        return logits

    # ===========================================================
    def inference_temporal(self, video, au_prompts):
        """
        Inference path — identical to training, except with no grad.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_temporal(video, au_prompts)
        return logits


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, num_aus=46, num_classes=2, au_prompts=None, learned_cls=False, is_video_clip=False, frame_stride=1, key_frames=16):
    if test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    elif 'bah' in test_set:
        classnames = bahssub_classes
    else:
        classnames = biosub_classes 

    if is_video_clip:
        model = ClipTestTimeVideoTuning(device, classnames, None, au_prompts, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_aus=num_aus, num_classes=num_classes,
                            text_hidden=512, clf_hidden=256, 
                            d_model=512, num_layers=4, nhead=8, dim_forward=2048, delta_stride=1)
    else:
        model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_aus=num_aus, num_classes=num_classes)

    return model