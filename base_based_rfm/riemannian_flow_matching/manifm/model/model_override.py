import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifm.model.arch import ACTFNS
import manifm.model.diffeq_layers as diffeq_layers


NormLayer = nn.RMSNorm


class FlowModelOverride(nn.Module):
    """A wrapper to override the forward method of the original model with conditional inputs."""

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
        actfn = ACTFNS[actfn]

        self.stem = Stem(
            in_dim,
            hidden_dim,
            actfn,
            embed_dim=embed_dim,
            num_classes=num_classes,
            null_chance=null_chance,
        )

        layers = []
        for _ in range(num_layers - 2):
            layers.append(LinearBlock(hidden_dim, hidden_dim, actfn, embed_dim, dropout=dropout))
        self.core = nn.ModuleList(layers)

        self.head = Head(hidden_dim, in_dim)

    def forward(self, t, x, y=None):
        x, y = self.stem(t, x, y=y)

        for layer in self.core:
            x = layer(t, x, y)

        x = self.head(t, x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, actfn, cond_dim, dropout=0.1, num_heads=8):
        super().__init__()
        self.norm1 = NormLayer(in_dim)
        self.ada_params = nn.Linear(cond_dim, 3 * in_dim)

        self.linear = diffeq_layers.ConcatLinear_v2(in_dim, out_dim)
        self.actfn = actfn(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = NormLayer(out_dim)
        self.attn = CrossAttention(out_dim, cond_dim, num_heads=num_heads)

        self.attn_gate = nn.Linear(cond_dim, out_dim)

        torch.nn.init.zeros_(self.ada_params.weight)
        torch.nn.init.zeros_(self.ada_params.bias)
        torch.nn.init.zeros_(self.attn_gate.weight)
        torch.nn.init.zeros_(self.attn_gate.bias)

    def forward(self, t, x, y):
        gamma, beta, alpha = self.ada_params(y).chunk(3, dim=-1)

        residual = x
        x = self.norm1(x)
        x = x * (1 + gamma) + beta  # Modulate

        x = self.linear(t, x)
        x = self.actfn(t, x)
        x = self.dropout(x)

        x = residual + alpha * x

        attn_residual = x
        attn_out = self.attn(self.norm2(x), y)

        gamma_attn = self.attn_gate(y)

        return attn_residual + gamma_attn * attn_out


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim, actfn, embed_dim=64, num_classes=None, null_chance=0.0):
        super().__init__()
        if num_classes is None:
            if null_chance > 0:
                raise ValueError("Null chance > 0 is not compatible with num_classes=None")
            num_classes = 1

        self.null_chance = null_chance

        self.class_embed = nn.Embedding(num_classes + 1 if null_chance > 0 else num_classes, embed_dim)
        self.class_mlp = EmbeddingMLP(embed_dim, embed_dim)

        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.time_mlp = EmbeddingMLP(embed_dim, embed_dim)

        self.cond_mlp = EmbeddingMLP(embed_dim * 2, embed_dim)

        self.concat_fc = diffeq_layers.ConcatLinear_v2(embed_dim + in_dim, out_dim)
        self.norm = NormLayer(out_dim)
        self.actfn = actfn(out_dim)

    def forward(self, t, x, y):
        if y is None:
            y = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        else:
            y = y + 1

        if self.training and self.null_chance > 0:
            mask = torch.rand_like(y, dtype=torch.float) < self.null_chance
            y[mask] = 0

        y_emb = self.class_embed(y)
        y_emb = self.class_mlp(y_emb)

        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        cond_emb = torch.cat([t_emb, y_emb], dim=1)
        cond_emb = self.cond_mlp(cond_emb)

        x = torch.cat([x, cond_emb], dim=1)
        x = self.concat_fc(t, x)
        x = self.norm(x)
        x = self.actfn(t, x)
        return x, cond_emb


class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = NormLayer(in_dim)
        self.linear = diffeq_layers.ConcatLinear_v2(in_dim, out_dim)

    def forward(self, t, x):
        x = self.norm(x)
        x = self.linear(t, x)
        return x


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.to_film = nn.Linear(cond_dim, 2 * hidden_dim)
        torch.nn.init.normal_(self.to_film.weight, mean=0.0, std=0.5)
        torch.nn.init.zeros_(self.to_film.bias)

    def forward(self, x, cond):
        gamma, beta = self.to_film(cond).chunk(2, dim=-1)
        x = (1 + gamma) * x + beta
        return x


class EmbeddingMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2

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
        angles = 2 * math.pi * torch.outer(t_vec, self.freqs)

        torch.sin(angles, out=emb[:, :half_dim])
        torch.cos(angles, out=emb[:, half_dim:2*half_dim])

        if self.dim % 2 != 0:
            emb[:, -1] = 0.0

        return emb


class CrossAttention(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(cond_dim, dim * 2, bias=False)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, cond):
        B, D = x.shape
        q = self.to_q(x.unsqueeze(1))
        kv = self.to_kv(cond.unsqueeze(1))

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        kv = kv.view(B, 1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).reshape(B, D)
        return self.to_out(out)
