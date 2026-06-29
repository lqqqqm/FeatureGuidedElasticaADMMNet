from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        hw: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_hw = hw
        self.dim = dim
        self.pos_embed = nn.Parameter(torch.zeros(1, hw * hw, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def _resize_pos_embed(self, h: int, w: int) -> torch.Tensor:
        if h == self.base_hw and w == self.base_hw:
            return self.pos_embed
        pe = self.pos_embed.reshape(1, self.base_hw, self.base_hw, self.dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, self.dim)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self._resize_pos_embed(h, w)
        for blk in self.blocks:
            t = blk(t)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        x = self.refine(x)
        return x


class MaskAwareTransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.norm1(x)
        attn_out, _ = self.attn(
            x1,
            x1,
            x1,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MaskAwareTransformerBottleneck(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        hw: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_hw = hw
        self.dim = dim
        self.pos_embed = nn.Parameter(torch.zeros(1, hw * hw, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                MaskAwareTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def _resize_pos_embed(self, h: int, w: int) -> torch.Tensor:
        if h == self.base_hw and w == self.base_hw:
            return self.pos_embed
        pe = self.pos_embed.reshape(1, self.base_hw, self.base_hw, self.dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, self.dim)
        return pe

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self._resize_pos_embed(h, w)

        mask_tokens = mask.flatten(2).squeeze(1)
        key_padding_mask = mask_tokens == 0
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False

        for blk in self.blocks:
            t = blk(t, key_padding_mask=key_padding_mask)
        x = t.transpose(1, 2).reshape(b, c, h, w)
        x = self.refine(x)
        return x


class SemanticBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_hw: int = 32,
        transformer_dim: int = 512,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        use_mask_aware_transformer: bool = True,
    ):
        super().__init__()
        self.use_mask_aware_transformer = use_mask_aware_transformer
        self.enc1 = ConvBlock(in_channels, 64)
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(128, 128)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc3 = ConvBlock(256, 256)
        self.down3 = nn.Conv2d(256, transformer_dim, kernel_size=4, stride=2, padding=1)

        transformer_cls = MaskAwareTransformerBottleneck if use_mask_aware_transformer else TransformerBottleneck
        self.transformer = transformer_cls(
            dim=transformer_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            hw=base_hw,
            dropout=transformer_dropout,
        )

        self.up3_conv = nn.Conv2d(transformer_dim, 256, kernel_size=3, padding=1)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec1 = ConvBlock(64 + 64, 64)

    def forward(self, x: torch.Tensor):
        mask = x[:, 3:4, :, :]
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        x_b = self.down3(e3)
        if self.use_mask_aware_transformer:
            mask_b = F.interpolate(mask, size=x_b.shape[-2:], mode="nearest")
            b = self.transformer(x_b, mask_b)
        else:
            b = self.transformer(x_b)

        d3 = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=False)
        d3 = self.up3_conv(d3)
        f3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(f3, scale_factor=2.0, mode="bilinear", align_corners=False)
        d2 = self.up2_conv(d2)
        f2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(f2, scale_factor=2.0, mode="bilinear", align_corners=False)
        d1 = self.up1_conv(d1)
        f1 = self.dec1(torch.cat([d1, e1], dim=1))

        return f1, f2, f3
