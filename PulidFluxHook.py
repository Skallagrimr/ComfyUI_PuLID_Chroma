import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.flux.layers import timestep_embedding
import comfy
from .patch_util import PatchKeys

def is_chroma_model(model):
    """Detect if the model is Chroma or Flux-based."""
    is_chroma = (model.__class__.__name__ == "Chroma") or hasattr(model, "distilled_guidance_layer")
    
    # Log detection results for debugging
    model_class = model.__class__.__name__
    has_distilled_guidance = hasattr(model, "distilled_guidance_layer")
    
    if is_chroma:
        print(f"[PuLID-Chroma] ✅ Chroma model detected:")
        print(f"  - Model class: {model_class}")
        print(f"  - Has distilled_guidance_layer: {has_distilled_guidance}")
        print(f"  - Using Chroma-specific forward function")
    else:
        print(f"[PuLID-Chroma] ℹ️ Flux model detected:")
        print(f"  - Model class: {model_class}")
        print(f"  - Has distilled_guidance_layer: {has_distilled_guidance}")
        print(f"  - Using Flux-specific forward function")
    
    return is_chroma

def set_model_dit_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"]

    if "dit" not in to["patches_replace"]:
        to["patches_replace"]["dit"] = {}
    else:
        to["patches_replace"]["dit"] = to["patches_replace"]["dit"]

    if key not in to["patches_replace"]["dit"]:
        if "double_block" in key:
            if key == ("double_block", 18):
                to["patches_replace"]["dit"][key] = LastDitDoubleBlockReplace(pulid_patch, **patch_kwargs)
            else:
                to["patches_replace"]["dit"][key] = DitDoubleBlockReplace(pulid_patch, **patch_kwargs)
        else:
            to["patches_replace"]["dit"][key] = DitSingleBlockReplace(pulid_patch, **patch_kwargs)
        # model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["dit"][key].add(pulid_patch, **patch_kwargs)

def pulid_patch(img, pulid_model=None, ca_idx=None, weight=1.0, embedding=None, mask=None, transformer_options={}):
    pulid_img = weight * pulid_model.model.pulid_ca[ca_idx].to(img.device)(embedding, img)
    if mask is not None:
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        latent_image_shape = pulid_temp_attrs.get("latent_image_shape", None)
        if latent_image_shape is not None:
            bs, c, h, w = latent_image_shape
            mask = comfy.sampler_helpers.prepare_mask(mask, (bs, c, h, w), img.device)
            patch_size = transformer_options[PatchKeys.running_net_model].patch_size
            mask = comfy.ldm.common_dit.pad_to_patch_size(mask, (patch_size, patch_size))
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # (b, seq_len, _) =>(b, seq_len, pulid.dim)
            mask = mask[..., 0].unsqueeze(-1).repeat(1, 1, pulid_img.shape[-1]).to(dtype=pulid_img.dtype)
            del patch_size, latent_image_shape

        pulid_img = pulid_img * mask

        del mask, pulid_temp_attrs

    return pulid_img

class DitDoubleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        out = extra_options["original_block"](input_args)
        img = out['img']
        temp_img = img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                img = img + callback(temp_img,
                                     pulid_model=self.kwargs[i]['pulid_model'],
                                     ca_idx=self.kwargs[i]['ca_idx'],
                                     weight=self.kwargs[i]['weight'],
                                     embedding=self.kwargs[i]['embedding'],
                                     mask = self.kwargs[i]['mask'],
                                     transformer_options=transformer_options
                                     )
        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, img

        return out


class LastDitDoubleBlockReplace(DitDoubleBlockReplace):
    def __call__(self, input_args, extra_options):
        out = super().__call__(input_args, extra_options)
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
        pulid_temp_attrs["double_blocks_txt"] = out['txt']
        return out

class DitSingleBlockReplace:
    def __init__(self, callback, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, input_args, extra_options):
        transformer_options = extra_options["transformer_options"]
        pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})

        out = extra_options["original_block"](input_args)

        sigma = pulid_temp_attrs["timesteps"][0].detach().cpu().item()
        img = out['img']
        txt = pulid_temp_attrs['double_blocks_txt']
        real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
        temp_img = real_img
        for i, callback in enumerate(self.callback):
            if self.kwargs[i]["sigma_start"] >= sigma >= self.kwargs[i]["sigma_end"]:
                real_img = real_img + callback(temp_img,
                                               pulid_model=self.kwargs[i]['pulid_model'],
                                               ca_idx=self.kwargs[i]['ca_idx'],
                                               weight=self.kwargs[i]['weight'],
                                               embedding=self.kwargs[i]['embedding'],
                                               mask=self.kwargs[i]['mask'],
                                               transformer_options = transformer_options,
                                               )

        img = torch.cat((txt, real_img), 1)

        out['img'] = img

        del temp_img, pulid_temp_attrs, sigma, transformer_options, real_img, img

        return out

def pulid_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    print(f"[PuLID-Chroma] 🔄 Executing Flux-specific forward function")
    print(f"  - Image shape: {img.shape}")
    print(f"  - Text shape: {txt.shape}")
    print(f"  - Timesteps: {timesteps.shape}")
    
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    transformer_options[PatchKeys.running_net_model] = self
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.double_blocks):
        # 0 -> 18
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        # 0 -> 37
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    del transformer_options[PatchKeys.running_net_model]

    return img


def pulid_forward_orig_chroma(
    self,
    img: Tensor,
    img_ids: Tensor,
    context: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    guidance: Tensor,
    control = None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    """Chroma-specific forward function that integrates PuLID as small bias to modulation."""
    print(f"[PuLID-Chroma] 🚀 Executing Chroma-specific forward function")
    print(f"  - Image shape: {img.shape}")
    print(f"  - Context shape: {context.shape}")
    print(f"  - Timesteps: {timesteps.shape}")
    print(f"  - Guidance: {guidance.shape}")
    
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or context.ndim != 3:
        raise ValueError("Input img and context tensors must have 3 dimensions.")

    transformer_options[PatchKeys.running_net_model] = self
    
    # Get PuLID identity features from transformer options
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    
    # Log PuLID features detection
    has_identity_features = 'pulid_identity_features' in pulid_temp_attrs
    has_weight = 'pulid_weight' in pulid_temp_attrs
    patch_count = len(patches_replace.get("dit", {}))
    
    print(f"[PuLID-Chroma] 🎭 PuLID Features Status:")
    print(f"  - Identity features present: {has_identity_features}")
    print(f"  - Weight present: {has_weight}")
    print(f"  - Active patches: {patch_count}")
    
    # 1) Recreate Chroma's modulation path using distilled guidance
    mod_index_length = 344
    distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
    distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

    modulation_index = timestep_embedding(
        torch.arange(mod_index_length, device=img.device), 32
    ).to(img.dtype).unsqueeze(0).repeat(img.shape[0], 1, 1)

    timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1) \
                           .unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype)

    input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)  # [B, 344, 64]
    mod_vectors = self.distilled_guidance_layer(input_vec)                  # [B, 344, H]
    
    print(f"[PuLID-Chroma] 🔧 Modulation vectors created:")
    print(f"  - Input vector shape: {input_vec.shape}")
    print(f"  - Modulation vectors shape: {mod_vectors.shape}")
    print(f"  - Modulation dimension (H): {mod_vectors.shape[-1]}")
    
    # 2) Add PuLID identity bias if available
    if has_identity_features:
        id_features = pulid_temp_attrs['pulid_identity_features']
        weight = pulid_temp_attrs.get('pulid_weight', 1.0)
        print(f"  - Identity features shape: {id_features.shape if hasattr(id_features, 'shape') else 'N/A'}")
        print(f"  - Weight value: {weight}")
        
        # Initialize safe projector if not exists
        H = mod_vectors.shape[-1]  # e.g., 3072 for Chroma
        if not hasattr(self, "pulid_lin"):
            E = id_features.shape[-1]  # e.g., 512 if InsightFace; 768 if EVA-CLIP
            print(f"[PuLID-Chroma] 🏗️ Initializing PuLID projector: {E} -> {H}")
            self.pulid_ln = torch.nn.LayerNorm(E, elementwise_affine=False).to(device=img.device, dtype=img.dtype)
            self.pulid_lin = torch.nn.Linear(E, H, bias=False).to(device=img.device, dtype=img.dtype)
            # Very small initialization to keep bias gentle
            torch.nn.init.normal_(self.pulid_lin.weight, mean=0.0, std=1e-3)
            print(f"  - Projector initialized with std=1e-3")
        
        # Project face features to modulation space
        def project_face_to_H(face_feat, H):
            x = self.pulid_ln(face_feat)                # B x E, zero-mean, unit-ish var
            x = self.pulid_lin(x)                       # B x H
            return x.to(dtype=img.dtype)
        
        # Create identity bias
        id_bias = project_face_to_H(id_features, H)        # B x H
        id_bias = id_bias.unsqueeze(1).expand(-1, 344, -1)  # B x 344 x H
        
        # Use small alpha for stable integration
        alpha = 0.03 * weight  # Start conservative, user can adjust weight
        print(f"[PuLID-Chroma] ➕ Adding identity bias with alpha={alpha:.4f}")
        
        # Add bias to modulation vectors
        mod_vectors_orig_std = mod_vectors.std().item()
        mod_vectors = mod_vectors + alpha * id_bias
        mod_vectors_new_std = mod_vectors.std().item()
        
        print(f"  - Original mod_vectors std: {mod_vectors_orig_std:.4f}")
        print(f"  - New mod_vectors std: {mod_vectors_new_std:.4f}")
        print(f"  - Std ratio: {mod_vectors_new_std/mod_vectors_orig_std:.3f}")
        
        if mod_vectors_new_std / mod_vectors_orig_std > 2.0:
            print(f"  - ⚠️ Warning: Large std increase, consider reducing weight")
    else:
        print(f"  - No identity features found, running without PuLID injection")
    
    # 3) Use Chroma's native get_modulations to create proper structure
    print(f"[PuLID-Chroma] 🎯 Using Chroma's native get_modulations")
    mods = self.get_modulations(mod_vectors)
    print(f"  - Modulations type: {type(mods)}")
    
    # 4) Process text and image inputs
    txt = self.txt_in(context)
    img = self.img_in(img)
    
    print(f"[PuLID-Chroma] 📐 After input processing:")
    print(f"  - txt shape: {txt.shape}")
    print(f"  - img shape: {img.shape}")
    
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})

    # 5) Process double blocks using Chroma's native structure
    for i, block in enumerate(self.double_blocks):
        # Get the proper modulation for this block from Chroma's get_modulations
        img_mod, txt_mod = mods
        vec = (img_mod[i], txt_mod[i])  # Use Chroma's slicing

        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                print(f"[PuLID-Chroma] 🔄 Block {i} with PuLID patches")
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            txt = out["txt"]
            img = out["img"]
        else:
            print(f"[PuLID-Chroma] 🔄 Block {i} direct call")
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = torch.cat((txt, img), 1)

    # 6) Process single blocks using Chroma's native structure  
    for i, block in enumerate(self.single_blocks):
        # Use Chroma's modulation structure for single blocks too
        # Single blocks typically use a simpler format from the modulation
        single_mod_idx = len(self.double_blocks) + i
        if single_mod_idx < len(mods[0]):  # Check bounds
            vec = mods[0][single_mod_idx]  # Use img modulation for single blocks
        else:
            # Fallback to last modulation if we run out
            vec = mods[0][-1]

        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask
                                                       },
                                                      {
                                                          "original_block": block_wrap,
                                                          "transformer_options": transformer_options
                                                      })
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    img = img[:, txt.shape[1]:, ...]

    # 7) Final layer - use Chroma's native final modulation
    print(f"[PuLID-Chroma] 🏁 Final layer processing")
    # Call Chroma's native final layer without custom modulation
    img = self.final_layer(img)

    del transformer_options[PatchKeys.running_net_model]

    print(f"[PuLID-Chroma] ✨ Chroma-specific forward completed successfully")
    print(f"  - Final output shape: {img.shape}")
    
    return img


def pulid_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    timesteps = timesteps.to(img.device, img.dtype)

    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['timesteps'] = timesteps

    transformer_options[PatchKeys.pulid_patch_key_attrs] = pulid_temp_attrs

    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options

def set_hook(m, new_forward):
    if hasattr(m, 'old_forward_orig_for_pulid'):
        return
    m.old_forward_orig_for_pulid = m.forward_orig
    m.forward_orig = new_forward.__get__(m)

def clean_hook(diffusion_model):
    # if hasattr(comfy.ldm.flux.model.Flux, 'old_forward_orig_for_pulid'):
    #     comfy.ldm.flux.model.Flux.forward_orig = comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    #     del comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    if hasattr(diffusion_model, 'old_forward_orig_for_pulid'):
        diffusion_model.forward_orig = diffusion_model.old_forward_orig_for_pulid
        del diffusion_model.old_forward_orig_for_pulid

def pulid_patch_double_blocks_after(img, txt, transformer_options):
    pulid_temp_attrs = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    pulid_temp_attrs['double_blocks_txt'] = txt
    return img, txt