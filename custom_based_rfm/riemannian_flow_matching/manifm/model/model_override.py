import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifm.model.arch import ACTFNS
import manifm.model.diffeq_layers as diffeq_layers

NormLayer = nn.RMSNorm


class FlowModelOverride(nn.Module):
    """A wrapper processing 3D tensors [B, S, D] preserving sequence layout."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        actfn,
        fourier=None,
        dropout=0.3,
        embed_dim=64,
        num_classes=None,
        null_chance=0.0,
    ):
        super().__init__()
        if fourier:
            raise NotImplementedError("Fourier features not implemented in override model.")

        self.is_cond = num_classes is not None and num_classes > 0
        actfn_cls = ACTFNS[actfn]

        self.stem = Stem(
            in_dim,
            hidden_dim,
            actfn_cls,
            embed_dim=embed_dim,
            num_classes=num_classes,
            null_chance=null_chance,
        )

        layers = []
        for _ in range(num_layers - 2):
            layers.append(LinearBlock(hidden_dim, actfn_cls, embed_dim, dropout=dropout))
        self.core = nn.ModuleList(layers)

        self.head = Head(hidden_dim, in_dim)

    def forward(self, t, x, y=None):
        # x incoming shape: [B, S, in_dim]
        B, S, D = x.shape

        # --- THE GLOBAL STRUCTURAL ALIGNMENT ---
        # Instead of matching 1D and 3D arrays back-and-forth across custom layers,
        # we tile the incoming time vector t to align completely with tokens [B * S].
        # This keeps activations and concat layers happy natively.
        if t.numel() == 1:
            t_flat = t.expand(B * S)
        elif t.numel() == B:
            t_flat = t.view(B, 1).expand(B, S).reshape(B * S)
        else:
            t_flat = t.flatten()

        # Stem expectations handled internally via flat timelines
        x, cond = self.stem(t_flat, x, y=y, B=B, S=S)

        # Loop processing over standard core blocks
        for layer in self.core:
            x = layer(t_flat, x, cond, B=B, S=S)

        # Reshape to token batch for the head projection layer
        hidden_dim = x.shape[-1]
        x_flat = x.reshape(B * S, hidden_dim)
        
        x_out_flat = self.head(t_flat, x_flat)
        return x_out_flat.view(B, S, -1)


class LinearBlock(nn.Module):
    def __init__(self, dim, actfn, cond_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = NormLayer(dim)
        self.ada_params = nn.Linear(cond_dim, 3 * dim)

        self.linear = diffeq_layers.ConcatLinear_v2(dim, dim)
        self.actfn = actfn(dim)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = NormLayer(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.attn_gate = nn.Linear(cond_dim, dim)

        torch.nn.init.normal_(self.ada_params.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.to_out.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.attn_gate.weight, mean=0.0, std=0.01)

        torch.nn.init.zeros_(self.ada_params.bias)
        torch.nn.init.zeros_(self.to_out.bias)
        torch.nn.init.zeros_(self.attn_gate.bias)

    def forward(self, t, x, cond, B, S):
        # x is flat [B * S, dim], cond is flat [B * S, cond_dim]
        D = x.shape[-1]

        # --- Stage 1: Adaptive Transform (AdaLN) ---
        gamma, beta, alpha = self.ada_params(cond).chunk(3, dim=-1)

        residual = x
        x = self.norm1(x)
        x = x * (1 + gamma) + beta  

        x = self.linear(t, x)
        x = self.actfn(t, x)
        x = self.dropout(x)
        x = residual + alpha * x

        # --- Stage 2: Isolated Sequence Attention ---
        # Temporarily reform 3D context blocks to compute attention paths
        x_3d = x.view(B, S, D)
        attn_residual = x_3d
        x_norm = self.norm2(x_3d)

        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(
            lambda tensor: tensor.view(B, S, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = self.to_out(attn_out)

        # Apply spatial structural gating adjustments
        gamma_attn = self.attn_gate(cond).view(B, S, D)
        x_3d = attn_residual + gamma_attn * attn_out

        # Flatten output back into 2D token arrays before returning
        return x_3d.view(B * S, D)


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim, actfn, embed_dim=64, num_classes=None, null_chance=0.0):
        super().__init__()
        if num_classes is None:
            if null_chance > 0:
                raise ValueError("Null chance > 0 is not compatible with num_classes=None")
            num_classes = 1

        self.null_chance = null_chance
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(num_classes + 1 if null_chance > 0 else num_classes, embed_dim)
        self.class_mlp = EmbeddingMLP(embed_dim, embed_dim)

        self.time_embed = SinusoidalEmbedding(embed_dim, angle_coverage=2 * torch.pi)
        self.time_mlp = EmbeddingMLP(embed_dim, embed_dim)

        self.cond_mlp = EmbeddingMLP(embed_dim * 2, embed_dim)

        self.pos_embed = SinusoidalEmbedding(embed_dim)
        self.pos_mlp = EmbeddingMLP(embed_dim, embed_dim)
        
        self.pos_gate = nn.Parameter(torch.tensor([0.01]))

        self.concat_fc = diffeq_layers.ConcatLinear_v2(embed_dim + in_dim, out_dim)
        self.norm = NormLayer(out_dim)
        self.actfn = actfn(out_dim)

    def forward(self, t, x, y, B, S):
        # t is flat [B * S], x is incoming 3D [B, S, in_dim]
        in_dim = x.shape[-1]

        # 1. Compute Global Conditions (Time + Class)
        if y is None:
            y = torch.zeros((B,), dtype=torch.long, device=x.device)
        else:
            y = y + 1

        if self.training and self.null_chance > 0:
            mask = torch.rand_like(y.float()) < self.null_chance
            y[mask] = 0

        y_emb = self.class_embed(y) # [B, embed_dim]
        y_emb = self.class_mlp(y_emb).unsqueeze(1).expand(-1, S, -1).reshape(B * S, self.embed_dim)

        # Obtain temporal projections utilizing unified flat layout
        t_emb = self.time_embed(t) # [B * S, embed_dim]
        t_emb = self.time_mlp(t_emb)

        cond_emb = torch.cat([t_emb, y_emb], dim=-1)
        cond_emb = self.cond_mlp(cond_emb) # [B * S, embed_dim]

        # 2. Compute and Inject Sequence Positional Encodings
        pos_indices = torch.arange(S, device=x.device)
        pos_emb = self.pos_embed(pos_indices) 
        pos_emb = self.pos_mlp(pos_emb).unsqueeze(0).expand(B, -1, -1).reshape(B * S, self.embed_dim)
        
        cond_emb = cond_emb + self.pos_gate * pos_emb

        # 3. Project to Output Feature Spaces
        x_flat = x.reshape(B * S, in_dim)
        x_features = torch.cat([x_flat, cond_emb], dim=-1)
        
        x_out = self.concat_fc(t, x_features)
        x_out = self.norm(x_out)
        x_out = self.actfn(t, x_out) # Now perfectly maps since both match [B * S]
        
        return x_out, cond_emb


class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = NormLayer(in_dim)
        self.linear = diffeq_layers.ConcatLinear_v2(in_dim, out_dim)

    def forward(self, t, x):
        # Both arrays match 2D token layouts [B * S] natively here
        x = self.norm(x)
        x = self.linear(t, x)
        return x


class EmbeddingMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()

        torch.nn.init.xavier_normal_(self.fc1.weight, gain=0.02)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=0.1)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, angle_coverage=1, max_period=10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.angle_coverage = angle_coverage

        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim).float()
            / (half_dim - 1)
        )
        self.register_buffer('freqs', freqs)

    @torch.no_grad()
    def forward(self, t):
        t_vec = t.reshape(-1).float()
        batch_size = t_vec.shape[0]
        half_dim = self.dim // 2

        emb = torch.empty((batch_size, self.dim), device=t.device, dtype=torch.float32)
        angles = self.angle_coverage * torch.outer(t_vec, self.freqs)

        torch.sin(angles, out=emb[:, :half_dim])
        torch.cos(angles, out=emb[:, half_dim:2*half_dim])

        if self.dim % 2 != 0:
            emb[:, -1] = 0.0

        return emb
