import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NormLayer = nn.RMSNorm

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='ij'), dim=0) 
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_value(embed_dim // 2, grid[0])  
    emb_w = get_1d_sincos_pos_embed_value(embed_dim // 2, grid[1])  
    emb = torch.cat([emb_h, emb_w], dim=1) 
    return emb

def get_1d_sincos_pos_embed_value(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim // 2)
    omega = 1. / (10000**omega)  

    pos = pos.reshape(-1)  
    out = torch.outer(pos, omega)  

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  
    return emb

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network to match state-of-the-art architectures."""
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class GatedAttentionBlock(nn.Module):
    """Scaled up Transformer block utilizing AdaLN-Single conditioning and SwiGLU."""
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = NormLayer(hidden_dim, elementwise_affine=False)
        self.norm2 = NormLayer(hidden_dim, elementwise_affine=False)
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
        hidden_mlp_dim = int(2 * (hidden_dim * 4) / 3)
        self.mlp = SwiGLU(hidden_dim, hidden_mlp_dim, dropout)
        
        self.attn_gate = nn.Parameter(torch.full((hidden_dim,), 0.02))
        self.mlp_gate = nn.Parameter(torch.full((hidden_dim,), 0.02))

    def forward(self, x: torch.Tensor, scale_shift_gate: tuple[torch.Tensor, ...]) -> torch.Tensor:
        B, N, _ = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = scale_shift_gate
        
        residual = x
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        
        qkv = self.qkv_proj(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, -1)
        attn_out = self.out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        
        x = residual + alpha1.unsqueeze(1) * (self.attn_gate * attn_out)
        
        residual = x
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        
        x = residual + alpha2.unsqueeze(1) * (self.mlp_gate * self.mlp(h))
        return x

class PatchStem(nn.Module):
    def __init__(self, img_channels: int, hidden_dim: int = 768, patch_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv2d(img_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = NormLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return self.dropout(self.norm(x)), H_out, W_out


class PixelHead(nn.Module):
    """High-fidelity reconstruction head using PixelShuffle to mitigate checkerboard artifacts."""
    def __init__(self, hidden_dim: int = 768, img_channels: int = 3, patch_size: int = 4):
        super().__init__()
        assert patch_size == 2 or patch_size == 4, "Designed specifically for standard patch dimensions."
        
        out_channels = img_channels * (patch_size ** 2)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )
        self.shuffle = nn.PixelShuffle(patch_size)
        self.final_act = nn.Tanh()

    def forward(self, x: torch.Tensor, H_grid: int, W_grid: int) -> torch.Tensor:
        B, N, D = x.shape
        x = x.permute(0, 2, 1).view(B, D, H_grid, W_grid)
        x = self.conv_block(x)
        return self.final_act(self.shuffle(x))


class ConditioningStem(nn.Module):
    def __init__(self, num_classes: int, cond_dim: int = 256, null_chance: float = 0.1):
        super().__init__()
        self.null_chance = null_chance
        self.class_embed = nn.Embedding(num_classes + 1, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, class_ids: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        y = class_ids + 1
        if self.training and self.null_chance > 0:
            mask = torch.rand_like(y, dtype=torch.float) < self.null_chance
            y[mask] = 0
            
        if force_drop:
            y = torch.zeros_like(y)
            
        return self.mlp(self.class_embed(y))

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 3, hidden_dim: int = 768, cond_dim: int = 256, 
                 num_classes: int = 10, num_layers: int = 12, patch_size: int = 4, 
                 img_channels: int = 3, dropout: float = 0.1, null_chance: float = 0.1, image_size: int = 32):
        super().__init__()
        self.cond_stem = ConditioningStem(num_classes, cond_dim, null_chance=null_chance)
        self.patch_stem = PatchStem(img_channels, hidden_dim, patch_size=patch_size, dropout=dropout)
        
        self.ada_single = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_layers * 6 * hidden_dim)
        )
        
        grid_size = image_size // patch_size
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(hidden_dim, grid_size, cls_token=False).unsqueeze(0))
        
        self.blocks = nn.ModuleList([
            GatedAttentionBlock(hidden_dim=hidden_dim, num_heads=12, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = NormLayer(hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, class_ids: torch.Tensor, force_drop: bool = False) -> tuple[torch.Tensor, int, int]:
        cond = self.cond_stem(class_ids, force_drop=force_drop)
        h, H_grid, W_grid = self.patch_stem(x)
        
        h = h + self.pos_embed.to(device=h.device, dtype=h.dtype)
        
        conditioning_chunks = self.ada_single(cond).chunk(len(self.blocks) * 6, dim=-1)
        
        for i, block in enumerate(self.blocks):
            block_cond = conditioning_chunks[i*6 : (i+1)*6]
            h = block(h, block_cond)
            
        h = self.final_norm(h)
        h = self.bottleneck(h)
        return F.normalize(h, p=2, dim=-1), H_grid, W_grid


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 3, hidden_dim: int = 768, cond_dim: int = 256, 
                 num_classes: int = 10, num_layers: int = 12, patch_size: int = 4, 
                 img_channels: int = 3, dropout: float = 0.1, null_chance: float = 0.1, image_size: int = 32):
        super().__init__()
        self.cond_stem = ConditioningStem(num_classes, cond_dim, null_chance=null_chance)
        self.input_mapping = nn.Linear(latent_dim, hidden_dim)
        
        self.ada_single = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_layers * 6 * hidden_dim)
        )
        
        grid_size = image_size // patch_size
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(hidden_dim, grid_size, cls_token=False).unsqueeze(0))
        
        self.blocks = nn.ModuleList([
            GatedAttentionBlock(hidden_dim=hidden_dim, num_heads=12, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = NormLayer(hidden_dim)
        self.pixel_head = PixelHead(hidden_dim, img_channels, patch_size=patch_size)

    def forward(self, z: torch.Tensor, class_ids: torch.Tensor, H_grid: int, W_grid: int, force_drop: bool = False) -> torch.Tensor:
        cond = self.cond_stem(class_ids, force_drop=force_drop)
        z_projected = F.normalize(z, p=2, dim=-1)
        h = self.input_mapping(z_projected)
        
        h = h + self.pos_embed.to(device=h.device, dtype=h.dtype)
        
        conditioning_chunks = self.ada_single(cond).chunk(len(self.blocks) * 6, dim=-1)
        
        for i, block in enumerate(self.blocks):
            block_cond = conditioning_chunks[i*6 : (i+1)*6]
            h = block(h, block_cond)
            
        h = self.final_norm(h)
        return self.pixel_head(h, H_grid, W_grid)


class SphericalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 3, hidden_dim: int = 768, cond_dim: int = 256, 
                 num_classes: int = 10, num_layers: int = 12, patch_size: int = 4, 
                 img_channels: int = 3, dropout: float = 0.1, null_chance: float = 0.1, image_size: int = 32):
        super().__init__()
        
        self.encoder = Encoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, 
            num_classes=num_classes, num_layers=num_layers, patch_size=patch_size, 
            img_channels=img_channels, dropout=dropout, null_chance=null_chance, image_size=image_size
        )
        self.decoder = Decoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, 
            num_classes=num_classes, num_layers=num_layers, patch_size=patch_size, 
            img_channels=img_channels, dropout=dropout, null_chance=null_chance, image_size=image_size
        )

    def forward(self, x: torch.Tensor, class_ids: torch.Tensor, 
                force_drop_enc: bool = False, force_drop_dec: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        
        latents, H_grid, W_grid = self.encoder(x, class_ids, force_drop=force_drop_enc)
        reconstructions = self.decoder(latents, class_ids, H_grid, W_grid, force_drop=force_drop_dec)
        
        return reconstructions, latents

if __name__ == "__main__":
    model = SphericalAutoencoder()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Scaled Parameters: {total_params / 1e6:.2f}M") # target ~98.4M
    
    x = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 10, (2,))
    
    recon, latent = model(x, labels)
    print("Output shapes -> Reconstruction:", recon.shape, "Latent:", latent.shape)
    
    magnitude = torch.norm(latent, p=2, dim=-1)
    print("Is Latent Spherical (magnitude == 1.0)?:", torch.allclose(magnitude, torch.ones_like(magnitude)))