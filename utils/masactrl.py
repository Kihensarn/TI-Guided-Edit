import torch
from einops import rearrange
from .masactrl_utils import AttentionBase

class UnifiedSelfAttentionControl(AttentionBase):
    def __init__(self, appearance_start_step=10, appearance_end_step=10, appearance_start_layer=10, 
                 struct_start_step=30, struct_end_step=30, struct_start_layer=8, mix_type="both", 
                 contrast_strength=1.67, injection_step=1):
        super().__init__()
        self.mix_type = mix_type
        self.contrast_strength = contrast_strength
        self.injection_step = injection_step
        self.appearance_start_step = appearance_start_step
        self.appearance_end_step = appearance_end_step
        self.appearance_start_layer = appearance_start_layer
        self.appearance_layer_idx = list(range(appearance_start_layer, 16))
        self.appearance_step_idx = list(range(appearance_start_step, appearance_end_step))

        self.struct_end_step = struct_end_step
        self.struct_start_step = struct_start_step
        self.struct_start_layer = struct_start_layer
        self.struct_layer_idx = list(range(struct_start_layer, 16))
        self.struct_step_idx = list(range(struct_start_step, struct_end_step))

        print("appearance step_idx: ", self.appearance_step_idx)
        print("appearance layer_idx: ", self.appearance_layer_idx)
        print("struct step_idx: ", self.struct_step_idx)
        print("struct layer_idx: ", self.struct_layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def contrast_attn(self, attn_map, contrast_factor):
        attn_mean = torch.mean(attn_map, dim=(0), keepdim=True)
        attn_map = (attn_map - attn_mean) * contrast_factor + attn_mean
        attn_map = torch.clip(attn_map, min=0.0, max=1.0)
        return attn_map

    def attn_batch_app(self, qc, kc, vc, ks, vs, sim, attn, num_heads, contrast_factor, is_rearrange, is_contrast, **kwargs):
        b = qc.shape[0] // num_heads
        qc = rearrange(qc, "(b h) n d -> h (b n) d", h=num_heads)
        kc = rearrange(kc, "(b h) n d -> h (b n) d", h=num_heads)
        vc = rearrange(vc, "(b h) n d -> h (b n) d", h=num_heads)
        ks = rearrange(ks, "(b h) n d -> h (b n) d", h=num_heads)
        vs = rearrange(vs, "(b h) n d -> h (b n) d", h=num_heads)
        sim_source = torch.einsum("h i d, h j d -> h i j", qc, kc) * kwargs.get("scale")
        sim_target = torch.einsum("h i d, h j d -> h i j", qc, ks) * kwargs.get("scale")

        if is_rearrange:
            v = torch.cat([vs, vc], dim=-2)
            C = torch.log2(torch.exp(sim_source).sum(dim=-1) / torch.exp(sim_target).sum(dim=-1))
            sim = torch.cat([sim_target+C.unsqueeze(-1), sim_source], dim=-1)
            attn = sim.softmax(-1)
        else:
            v = vs
            attn = sim_target.softmax(-1)

        if is_contrast:
            attn = self.contrast_attn(attn, contrast_factor)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def unified_attn_batch(self, qc, kc, vc, qs, ks, vs, qa, ka, va, sim, attn, contrast_factor, num_heads, **kwargs):
        b = qc.shape[0] // num_heads
        qc = rearrange(qc, "(b h) n d -> h (b n) d", h=num_heads)
        kc = rearrange(kc, "(b h) n d -> h (b n) d", h=num_heads)
        vc = rearrange(vc, "(b h) n d -> h (b n) d", h=num_heads)
        qs = rearrange(qs, "(b h) n d -> h (b n) d", h=num_heads)
        ks = rearrange(ks, "(b h) n d -> h (b n) d", h=num_heads)
        vs = rearrange(vs, "(b h) n d -> h (b n) d", h=num_heads)
        qa = rearrange(qa, "(b h) n d -> h (b n) d", h=num_heads)
        ka = rearrange(ka, "(b h) n d -> h (b n) d", h=num_heads)
        va = rearrange(va, "(b h) n d -> h (b n) d", h=num_heads)
        v = torch.cat([va, vc], dim=-2)

        sim_source = torch.einsum("h i d, h j d -> h i j", qc, kc) * kwargs.get("scale")
        sim_target_struct = torch.einsum("h i d, h j d -> h i j", qs, ks) * kwargs.get("scale")
        sim_target_app = torch.einsum("h i d, h j d -> h i j", qc, ka) * kwargs.get("scale")

        attn_target_struct = sim_target_struct.softmax(-1) 
        attn_target_struct = self.contrast_attn(attn_target_struct, contrast_factor)
        sim_target = torch.matmul(attn_target_struct, sim_target_app)

        C = torch.log2(torch.exp(sim_source).sum(dim=-1) / torch.exp(sim_target).sum(dim=-1))
        sim = torch.cat([sim_target+C.unsqueeze(-1), sim_source], dim=-1)
        attn = sim.softmax(-1)

        attn = self.contrast_attn(attn, contrast_factor)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # appearance
        out_u_0 = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_0 = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # struct
        out_u_2 = self.attn_batch(qu[num_heads*2:], ku[num_heads*2:], vu[num_heads*2:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_2 = self.attn_batch(qc[num_heads*2:], kc[num_heads*2:], vc[num_heads*2:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # target
        out_u_1 = self.attn_batch(qu[num_heads:num_heads*2], ku[num_heads:num_heads*2], vu[num_heads:num_heads*2], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_1 = self.attn_batch(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        
        # determine the mix type of each layer in each step
        if self.cur_step % self.injection_step == 0 and (self.cur_step in self.appearance_step_idx and self.cur_att_layer // 2 in self.appearance_layer_idx) and (self.cur_step in self.struct_step_idx and self.cur_att_layer // 2 in self.struct_layer_idx):
            cur_mix_type = "both"
        elif self.mix_type != "struct" and (self.cur_step in self.appearance_step_idx and self.cur_att_layer // 2 in self.appearance_layer_idx):
            cur_mix_type = "app"
        elif self.mix_type != "app" and (self.cur_step in self.struct_step_idx and self.cur_att_layer // 2 in self.struct_layer_idx):
            cur_mix_type = "struct"
        else:
            cur_mix_type = None

        # attention injection
        if cur_mix_type == "both":
            out_c_1 = self.unified_attn_batch(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], qc[num_heads*2:], kc[num_heads*2:], vc[num_heads*2:], qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, self.contrast_strength, num_heads, **kwargs)
        elif cur_mix_type == "app":
            out_c_1 = self.attn_batch_app(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], kc[:num_heads], vc[:num_heads], None, attnc, num_heads, self.contrast_strength, self.mix_type == "both", self.mix_type == "both", **kwargs)
        elif cur_mix_type == "struct":
            out_c_1 = self.attn_batch(qc[num_heads*2:], kc[num_heads*2:], vc[num_heads:num_heads*2], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            pass
                
        out = torch.cat([out_u_0, out_u_1, out_u_2, out_c_0, out_c_1, out_c_2], dim=0)
        return out