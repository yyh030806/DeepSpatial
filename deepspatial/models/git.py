import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from .commons import modulate, TimestepEmbedder, FinalLayer, get_1d_sincos_pos_embed

class PatchEmbedder(nn.Module):
    def __init__(self, input_size, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size + patch_size - 1) // patch_size
        
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        B, L = x.shape
        # Pad if not divisible
        pad_size = (self.patch_size - (L % self.patch_size)) % self.patch_size
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size), "constant", 0)
        
        # Reshape to [Batch, Num_Patches, Patch_Size]
        x = x.reshape(B, -1, self.patch_size) 
        x = self.mlp(x) 
        return x

class GiTBlock(nn.Module):
    """
    Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class GiT(nn.Module):
    """
    GiT: Separated x, g, c streams with physical depth and biological anchors.
    """
    def __init__(self, gene_dim, patch_size, hidden_size, depth, num_heads, num_classes, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        # x is 2D, g is gene_dim
        self.num_patches_x = 1 
        self.num_patches_g = math.ceil(gene_dim / patch_size)
        self.num_patches = self.num_patches_x + self.num_patches_g

        # 1. Input Embedders
        self.x_embedder = nn.Linear(2, hidden_size)
        self.g_embedder = PatchEmbedder(gene_dim, patch_size, hidden_size)
        
        # Condition Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = nn.Linear(num_classes, hidden_size)

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # 2. Transformer Backbone
        self.blocks = nn.ModuleList([
            GiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 3. Output Heads
        self.x_head = FinalLayer(hidden_size, 2, 1) # Spatial velocity
        self.g_head = FinalLayer(hidden_size, patch_size, 1) # Gene velocity
        self.c_head = nn.Linear(hidden_size, num_classes) # Cell type logits

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights using GiT/DiT standards (adaLN-Zero)."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize Positional Embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out modulation and final layers for identity mapping at start
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.x_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.x_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.x_head.linear.weight, 0)
        nn.init.constant_(self.x_head.linear.bias, 0)

        nn.init.constant_(self.g_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.g_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.g_head.linear.weight, 0)
        nn.init.constant_(self.g_head.linear.bias, 0)

    def forward(self, xt, gt, t, zt, delta_z, ct):
        """
        Forward pass with separated modalities.
        """
        # Embed modalities separately
        gene_dim = gt.shape[1]
        x_feat = self.x_embedder(xt).unsqueeze(1) # [B, 1, D]
        g_feat = self.g_embedder(gt) # [B, num_patches_g, D]

        h = torch.cat([x_feat, g_feat], dim=1) + self.pos_embed
        # Global conditioning
        cond = self.t_embedder(t.view(-1)) + \
               self.z_embedder(zt.view(-1)) + self.z_embedder(delta_z.view(-1)) + \
               self.c_embedder(ct)

        for block in self.blocks:
            h = block(h, cond)

        # Project back to modality-specific outputs
        x = self.x_head(h[:, :1, :], cond).squeeze(1) # [B, 2]
        g = self.g_head(h[:, 1:, :], cond).reshape(xt.shape[0], -1)[:, :gene_dim] # [B, Gene_Dim]
        c = self.c_head(h.mean(dim=1)) # [B, Classes]

        return x, g, c