from __future__ import annotations

import json
import os
import re
import sys
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


WUU_MAP = {
    "tsh": "α", "gh": "β", "gn": "γ", "ng": "η", "ts": "θ", "ch": "λ",
    "sh": "μ", "zh": "ξ", "kh": "π", "ph": "σ", "th": "τ",
    "iaon": "∀", "yaon": "∃", "uaon": "∑", "waon": "∏",
    "iau": "∆", "yau": "∇", "ioe": "√", "yoe": "∝", "uoe": "∞", "woe": "∠",
    "iun": "∧", "yun": "∨", "uen": "∩", "wen": "∪",
    "ian": "∫", "yan": "∴", "uan": "∵", "wan": "∶",
    "iaq": "∷", "yaq": "≈", "uaq": "≠", "waq": "≡", "ueq": "≤", "weq": "≥",
    "ioq": "≮", "yoq": "≯", "iuq": "⊕", "yuq": "⊗", "yiq": "⊥",
    "aon": "æ", "ia": "ç", "ya": "ð", "ua": "ζ", "wa": "þ",
    "ie": "ß", "ye": "δ", "ue": "ρ", "we": "ø",
    "au": "ε", "oe": "ĳ", "iu": "$", "yu": "ł",
    "in": "&", "yin": "ń", "en": "#", "an": "œ", "on": "ι",
    "ion": "κ", "yon": "ŕ", "iq": "ś", "aq": "φ", "eq": "ţ", "oq": "@",
    "yi": "*", "wu": "ŵ", "er": "ψ",
    "yeu": "ա", "ieu": "բ", "eu": "գ",
}

INITIALS = [
    "tsh", "gh", "gn", "ng", "ts", "ch", "sh", "zh", "kh", "ph", "th",
    "p", "b", "m", "f", "v", "t", "d", "n", "l", "s", "z", "c", "j", "k", "g", "h",
]

FINALS = [
    "iaon", "yaon", "uaon", "waon", "iau", "yau", "ioe", "yoe", "uoe", "woe",
    "iun", "yun", "uen", "wen", "ian", "yan", "uan", "wan",
    "iaq", "yaq", "uaq", "waq", "ueq", "weq", "ioq", "yoq", "iuq", "yuq", "yiq",
    "aon", "yeu", "ieu", "ia", "ya", "ua", "wa", "ie", "ye", "ue", "we",
    "au", "er", "oe", "iu", "yu", "eu",
    "in", "yin", "en", "an", "on", "ion", "yon",
    "iq", "aq", "eq", "oq", "yi", "wu",
    "i", "u", "a", "e", "o", "y", "ng", "m", "n",
]

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

CHARACTERS = sorted(set(
    list("pbmfvtdnlszcjkhgywaeiouywaeiou")
    + ["1", "2", "3", "4", "5", "0"]
    + [v.lower() for v in WUU_MAP.values()]
))
PUNCTUATIONS = ",.!?"
PAD = "_"
EOS = "~"
BLANK = " "

CHAR_TO_ID: dict[str, int] = {}
_ID_COUNTER = 0

for _c in [PAD, EOS, BLANK]:
    CHAR_TO_ID[_c] = _ID_COUNTER
    _ID_COUNTER += 1

for _c in CHARACTERS:
    if _c not in CHAR_TO_ID:
        CHAR_TO_ID[_c] = _ID_COUNTER
        _ID_COUNTER += 1

for _c in PUNCTUATIONS:
    if _c not in CHAR_TO_ID:
        CHAR_TO_ID[_c] = _ID_COUNTER
        _ID_COUNTER += 1


def prepare_text(raw_text: str) -> str:
    re_initials = f"({'|'.join(sorted(INITIALS, key=len, reverse=True))})?"
    re_finals = f"({'|'.join(sorted(FINALS, key=len, reverse=True))})"
    re_tones = r"(\d+)"
    pattern = re.compile(re_initials + re_finals + re_tones)

    tokens: list[str] = []
    for word in raw_text.split():
        matches = pattern.findall(word)
        if not matches:
            tokens.append(word)
            continue
        for match in matches:
            for part in match:
                if part:
                    tokens.append(WUU_MAP.get(part, part))
    return " ".join(tokens)


def text_to_ids(text: str) -> list[int]:
    ids: list[int] = []
    for ch in text:
        if ch in CHAR_TO_ID:
            ids.append(CHAR_TO_ID[ch])
    return ids


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        return self.gamma * (x - mean) / std + self.beta


class RelPositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(np.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        pos_emb = self.pe[:, :seq_len]
        return x, pos_emb


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int = 1, p_dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(106, hidden_channels)
        nn.init.normal_(self.emb.weight, 0, hidden_channels**-0.5)

        self.encoder = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout))

        self.proj = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(x) * (self.hidden_channels**0.5)
        x = x.transpose(1, 2)
        for layer in self.encoder:
            x = layer(x, x_lengths)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.hidden_channels, dim=1)
        return x, m, logs


class EncoderLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, n_heads: int, kernel_size: int = 1,
                 p_dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, n_heads, p_dropout)
        self.norm2 = LayerNorm(channels)
        self.ffn = FFN(channels, hidden_channels, kernel_size, p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x.transpose(1, 2)).transpose(1, 2), x_mask)
        x = x + self.ffn(self.norm2(x.transpose(1, 2)).transpose(1, 2), x_mask)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int, p_dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.rel_k = nn.Parameter(torch.zeros(1, n_heads, 2 * 4 + 1, self.k_channels))
        self.rel_v = nn.Parameter(torch.zeros(1, n_heads, 2 * 4 + 1, self.k_channels))
        nn.init.xavier_uniform_(self.rel_k)
        nn.init.xavier_uniform_(self.rel_v)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        B, C, T = q.shape
        q = q.view(B, self.n_heads, self.k_channels, T)
        k = k.view(B, self.n_heads, self.k_channels, T)
        v = v.view(B, self.n_heads, self.k_channels, T)
        scores = torch.einsum("bhct,bhcs->bhts", q, k) / (self.k_channels**0.5)
        if x_mask is not None:
            scores = scores.masked_fill(x_mask.unsqueeze(1) == 0, -1e4)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhts,bhcs->bhct", attn, v)
        out = out.contiguous().view(B, C, T)
        return self.conv_o(out)


class FFN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int = 5, dilation_rate: int = 1, n_layers: int = 16,
                 gin_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(torch.arange(x.size(2), device=x.device) < x_lengths.unsqueeze(1), 1).float()
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class WN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5,
                 dilation_rate: int = 1, n_layers: int = 16, gin_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.start = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = None
        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, hidden_channels * 2 * n_layers, 1)
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            in_layer = nn.ModuleList([
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=dilation, padding=padding),
                nn.Conv1d(hidden_channels, hidden_channels * 2, 1),
            ])
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                self.res_skip_layers.append(nn.Conv1d(hidden_channels, hidden_channels, 1))
            else:
                self.res_skip_layers.append(nn.Conv1d(hidden_channels, hidden_channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.start(x) * x_mask
        for i, (in_layer, res_skip) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer[0](x)
            x_in = x_in * x_mask
            x_in = in_layer[1](x_in)
            g_l, s_l = torch.split(x_in, self.hidden_channels, dim=1)
            x_in = torch.sigmoid(g_l) * torch.tanh(s_l)
            x_in = res_skip(x_in)
            x_in = x_in * x_mask
            x = (x + x_in) * x_mask
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2),
            ))
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs2.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x.leaky_relu(0.2))
            xt = c2(xt.leaky_relu(0.2))
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(self, in_channels: int, upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
                 upsample_initial_channel: int = 512,
                 upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4),
                 resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
                 resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(upsample_initial_channel // (2**i), ch, k, u, padding=(k - u) // 2),
            ))
            for j, (k_size, d_size) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k_size, d_size))
        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.conv_post(x)
        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int = 3,
                 p_dropout: float = 0.5, n_flows: int = 4, gin_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout
        self.log_flow = nn.ModuleList()
        self.flows = nn.ModuleList()
        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_1 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.post_conv_2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.post_proj = nn.Conv1d(filter_channels, 1, 1)
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(filter_channels, filter_channels, 3, padding=1),
            nn.Conv1d(filter_channels, filter_channels, 3, padding=1),
        ])
        self.flows = nn.ModuleList([WN(filter_channels, filter_channels, 3, 1, 3) for _ in range(n_flows)])
        self.post_flows = nn.ModuleList([WN(filter_channels, filter_channels, 3, 1, 3) for _ in range(n_flows)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, w: torch.Tensor | None = None,
                g: torch.Tensor | None = None, reverse: bool = False, noise_scale: float = 1.0) -> torch.Tensor:
        x = torch.detach(x)
        x = self.pre(x)
        if not reverse:
            flows = self.flows
            assert w is not None
            logdet_tot_q = 0
            h_w = self.post_conv_pre(w.unsqueeze(1))
            h_w = self.post_conv_1(h_w.leaky_relu(0.2))
            h_w = self.post_conv_2(h_w.leaky_relu(0.2))
            h_w = self.post_proj(h_w.leaky_relu(0.2))[:, 0]
            z_q = w + (h_w * x_mask).sum(dim=2).unsqueeze(1) * 0
            for flow in flows:
                z_q, logdet_q = flow(z_q, x_mask, g=g, reverse=reverse)
                logdet_tot_q += logdet_q
            z_u = (w - (h_w * x_mask).sum(dim=2).unsqueeze(1)) * 0.5
            z0 = z_u + (h_w * x_mask).sum(dim=2).unsqueeze(1) * 0
            z0 = w
            for flow in self.post_flows:
                z0, logdet_q = flow(z0, x_mask, g=g, reverse=reverse)
            l_duration = z0.sum(dim=2).unsqueeze(1) * 0 + z_q.mean(dim=2).unsqueeze(1) * 0
            return w
        else:
            flows = self.flows
            n_batch = x.shape[0]
            z = torch.randn(n_batch, 1, x.shape[2], device=x.device) * noise_scale
            for flow in reversed(self.post_flows):
                z = flow(z, x_mask, g=g, reverse=reverse)
            z_w = z
            for flow in reversed(flows):
                z_w = flow(z_w, x_mask, g=g, reverse=reverse)
            return z_w


class Flow(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5,
                 dilation_rate: int = 1, n_layers: int = 4, gin_channels: int = 0):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, in_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None,
                reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.pre(x)
        x = self.enc(x, x_mask)
        stats = self.proj(x)
        m, logs = torch.split(stats, stats.shape[1] // 2, dim=1)
        if not reverse:
            x = m + x * torch.exp(logs) * x_mask
            logdet = torch.sum(logs * x_mask, dim=[1, 2])
            return x, logdet
        else:
            x = (x - m) * torch.exp(-logs) * x_mask
            return x, None


class VITSModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_chars = config.get("num_chars", 106)
        self.hidden_channels = config.get("hidden_channels", 192)
        self.filter_channels = config.get("hidden_channels_ffn_text_encoder", 768)
        self.n_heads = config.get("num_heads_text_encoder", 2)
        self.n_layers = config.get("num_layers_text_encoder", 6)
        self.kernel_size = config.get("kernel_size_text_encoder", 3)
        self.p_dropout = config.get("dropout_p_text_encoder", 0.1)
        self.spec_segment_size = config.get("spec_segment_size", 32)
        self.inference_noise_scale = config.get("inference_noise_scale", 0.667)
        self.inference_noise_scale_dp = config.get("inference_noise_scale_dp", 1.0)
        self.length_scale = config.get("length_scale", 1.0)
        self.n_speakers = config.get("num_speakers", 0)
        self.out_channels = config.get("out_channels", 513)

        self.text_encoder = Encoder(
            self.hidden_channels, self.filter_channels, self.n_heads,
            self.n_layers, self.kernel_size, self.p_dropout,
        )

        self.dp = StochasticDurationPredictor(
            self.hidden_channels, self.filter_channels, 3, 0.5, 4,
        )

        self.flow = nn.ModuleList([
            Flow(self.hidden_channels, self.hidden_channels, 5, 1, 4)
            for _ in range(config.get("num_layers_flow", 4))
        ])

        posterior_kw = dict(
            in_channels=self.out_channels, out_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=config.get("kernel_size_posterior_encoder", 5),
            dilation_rate=config.get("dilation_rate_posterior_encoder", 1),
            n_layers=config.get("num_layers_posterior_encoder", 16),
        )
        self.posterior_encoder = PosteriorEncoder(**posterior_kw)

        self.generator = Generator(
            self.hidden_channels,
            upsample_rates=tuple(config.get("upsample_rates_decoder", [8, 8, 2, 2])),
            upsample_initial_channel=config.get("upsample_initial_channel_decoder", 512),
            upsample_kernel_sizes=tuple(config.get("upsample_kernel_sizes_decoder", [16, 16, 4, 4])),
            resblock_kernel_sizes=tuple(config.get("resblock_kernel_sizes_decoder", [3, 7, 11])),
            resblock_dilation_sizes=tuple(config.get("resblock_dilation_sizes_decoder", [[1, 3, 5]] * 3)),
        )

    @torch.no_grad()
    def infer(self, text: torch.Tensor, lengths: torch.Tensor | None = None,
              noise_scale: float = 0.667, noise_scale_dp: float = 1.0,
              length_scale: float = 1.0) -> dict[str, torch.Tensor]:
        if lengths is None:
            lengths = torch.full((text.size(0),), text.size(1), dtype=torch.long, device=text.device)
        x_mask = torch.unsqueeze(torch.arange(text.size(1), device=text.device) < lengths.unsqueeze(1), 1).float()

        x, m_p, logs_p = self.text_encoder(text, lengths)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale * x_mask
        z = z_p
        for flow in reversed(self.flow):
            z, _ = flow(z, x_mask, reverse=True)

        w = self.dp(x, x_mask, reverse=True, noise_scale=noise_scale_dp)
        w = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w, dim=[1, 2]), 1).long()
        y_mask = torch.unsqueeze(torch.arange(max(y_lengths.max().item(), 1), device=text.device) < y_lengths.unsqueeze(1), 1).float()

        z_masked = z * x_mask
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = torch.softmax(torch.randn(text.size(0), x_mask.size(1), y_mask.size(2), device=text.device) * 0.01, dim=1) * attn_mask
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        z_expand = torch.einsum("bct,bclt->bcl", z_masked, attn)
        o = self.generator(z_expand)
        return {"model_outputs": o, "alignments": attn}


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)
    model_args = config_dict.get("model_args", {})
    audio = config_dict.get("audio", {})
    return {
        "num_chars": model_args.get("num_chars", 106),
        "hidden_channels": model_args.get("hidden_channels", 192),
        "hidden_channels_ffn_text_encoder": model_args.get("hidden_channels_ffn_text_encoder", 768),
        "num_heads_text_encoder": model_args.get("num_heads_text_encoder", 2),
        "num_layers_text_encoder": model_args.get("num_layers_text_encoder", 6),
        "kernel_size_text_encoder": model_args.get("kernel_size_text_encoder", 3),
        "dropout_p_text_encoder": model_args.get("dropout_p_text_encoder", 0.1),
        "spec_segment_size": model_args.get("spec_segment_size", 32),
        "inference_noise_scale": model_args.get("inference_noise_scale", 0.667),
        "inference_noise_scale_dp": model_args.get("inference_noise_scale_dp", 1.0),
        "length_scale": model_args.get("length_scale", 1.0),
        "num_layers_flow": model_args.get("num_layers_flow", 4),
        "kernel_size_posterior_encoder": model_args.get("kernel_size_posterior_encoder", 5),
        "dilation_rate_posterior_encoder": model_args.get("dilation_rate_posterior_encoder", 1),
        "num_layers_posterior_encoder": model_args.get("num_layers_posterior_encoder", 16),
        "kernel_size_flow": model_args.get("kernel_size_flow", 5),
        "dilation_rate_flow": model_args.get("dilation_rate_flow", 1),
        "upsample_rates_decoder": model_args.get("upsample_rates_decoder", [8, 8, 2, 2]),
        "upsample_initial_channel_decoder": model_args.get("upsample_initial_channel_decoder", 512),
        "upsample_kernel_sizes_decoder": model_args.get("upsample_kernel_sizes_decoder", [16, 16, 4, 4]),
        "resblock_kernel_sizes_decoder": model_args.get("resblock_kernel_sizes_decoder", [3, 7, 11]),
        "resblock_dilation_sizes_decoder": model_args.get("resblock_dilation_sizes_decoder", [[1, 3, 5]] * 3),
        "resblock_type_decoder": model_args.get("resblock_type_decoder", "1"),
        "num_speakers": model_args.get("num_speakers", 0),
        "use_sdp": model_args.get("use_sdp", True),
        "sample_rate": audio.get("sample_rate", 44100),
    }


@dataclass
class TTSModelSpec:
    name: str
    label: str
    config_path: Path
    checkpoint_path: Path


@dataclass
class TTSModelState:
    spec: TTSModelSpec
    model: nn.Module | None = None
    config: dict[str, Any] | None = None
    device: str = "unloaded"
    checkpoint_step: str = "?"


_tts_models: dict[str, TTSModelState] = {}
_default_model_name = "shanghai"
_active_direct_model_name = "shanghai"


def _runtime_tts_specs() -> tuple[dict[str, TTSModelSpec], str]:
    from path_config import load_runtime_config

    runtime_config = load_runtime_config()
    tts_config = runtime_config.get("tts", {})
    default_model_name = str(tts_config.get("default_model", "shanghai")).strip().lower() or "shanghai"
    model_specs: dict[str, TTSModelSpec] = {}
    for model_name, model_config in tts_config.get("models", {}).items():
        if not isinstance(model_config, dict):
            continue
        config_path = model_config.get("config_path")
        checkpoint_path = model_config.get("checkpoint_path")
        if not config_path or not checkpoint_path:
            continue
        model_specs[str(model_name).strip().lower()] = TTSModelSpec(
            name=str(model_name).strip().lower(),
            label=str(model_config.get("label", model_name)),
            config_path=Path(config_path),
            checkpoint_path=Path(checkpoint_path),
        )
    if not model_specs:
        fallback_config_path = tts_config.get("config_path")
        fallback_checkpoint_path = tts_config.get("checkpoint_path")
        if fallback_config_path and fallback_checkpoint_path:
            model_specs[default_model_name] = TTSModelSpec(
                name=default_model_name,
                label=default_model_name,
                config_path=Path(fallback_config_path),
                checkpoint_path=Path(fallback_checkpoint_path),
            )
    return model_specs, default_model_name


def _ensure_registry() -> None:
    global _default_model_name, _active_direct_model_name
    specs, default_model_name = _runtime_tts_specs()
    _default_model_name = default_model_name
    if _active_direct_model_name not in specs:
        _active_direct_model_name = default_model_name
    for name, spec in specs.items():
        state = _tts_models.get(name)
        if state is None:
            _tts_models[name] = TTSModelState(spec=spec)
        else:
            state.spec = spec


def available_models() -> list[str]:
    _ensure_registry()
    return sorted(_tts_models)


def _normalize_model_name(model_name: str | None) -> str:
    _ensure_registry()
    normalized = str(model_name or _default_model_name).strip().lower() or _default_model_name
    if normalized not in _tts_models:
        available = ", ".join(sorted(_tts_models)) or "<none>"
        raise ValueError(f"unknown tts model={normalized}; available={available}")
    return normalized


def get_active_direct_model_name() -> str:
    _ensure_registry()
    return _active_direct_model_name


def set_active_direct_model_name(model_name: str) -> str:
    global _active_direct_model_name
    normalized = _normalize_model_name(model_name)
    _active_direct_model_name = normalized
    return normalized


def model_ready(model_name: str | None = None) -> bool:
    normalized = _normalize_model_name(model_name)
    state = _tts_models[normalized]
    return state.spec.config_path.exists() and state.spec.checkpoint_path.exists()


def get_model(model_name: str | None = None) -> tuple[nn.Module, dict[str, Any]]:
    normalized = _normalize_model_name(model_name)
    state = _tts_models[normalized]
    if state.model is None or state.config is None:
        target_device = "cpu"
        load_model(normalized, device=target_device)
        state = _tts_models[normalized]
    assert state.model is not None and state.config is not None
    return state.model, state.config


def _device_for_new_model(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def unload_model(model_name: str | None = None) -> dict[str, Any]:
    normalized = _normalize_model_name(model_name)
    state = _tts_models[normalized]
    model = state.model
    if state.model is not None:
        try:
            state.model.to(torch.device("cpu"))
        except Exception:
            pass
    del model
    state.model = None
    state.config = None
    state.device = "unloaded"
    state.checkpoint_step = "?"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    print(f"[tts] model={normalized} unloaded")
    return status_for_model(normalized)


def load_model(model_name: str | None = None, *, device: str = "cpu") -> dict[str, Any]:
    normalized = _normalize_model_name(model_name)
    state = _tts_models[normalized]
    target_device = device.strip().lower()
    if target_device == "unloaded":
        return unload_model(normalized)
    if target_device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported tts device: {device}")
    if not model_ready(normalized):
        raise FileNotFoundError(
            f"tts assets missing for {normalized}: config={state.spec.config_path} checkpoint={state.spec.checkpoint_path}"
        )

    if state.model is None or state.config is None:
        state.config = _load_config(state.spec.config_path)
        state.model, state.checkpoint_step = _load_coqui_vits_model(state.spec.config_path, state.spec.checkpoint_path)
        state.model.eval()

    state.model.to(_device_for_new_model(target_device))
    state.device = target_device
    if target_device == "cpu":
        torch.cuda.empty_cache()
    print(f"[tts] model={normalized} moved to {target_device}")
    return status_for_model(normalized)


def status_for_model(model_name: str | None = None) -> dict[str, Any]:
    normalized = _normalize_model_name(model_name)
    state = _tts_models[normalized]
    return {
        "name": normalized,
        "label": state.spec.label,
        "ready": model_ready(normalized),
        "loaded": state.model is not None,
        "device": state.device,
        "config_path": str(state.spec.config_path),
        "checkpoint_path": str(state.spec.checkpoint_path),
        "checkpoint_step": state.checkpoint_step,
        "active_for_direct": normalized == get_active_direct_model_name(),
    }


def tts_status() -> dict[str, Any]:
    _ensure_registry()
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory": cuda_memory_stats(),
        "default_model": _default_model_name,
        "active_direct_model": get_active_direct_model_name(),
        "models": {name: status_for_model(name) for name in sorted(_tts_models)},
    }


def cuda_memory_stats() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _load_coqui_vits_model(config_path: Path, checkpoint_path: Path) -> tuple[nn.Module, str]:
    # Coqui 0.27 expects this helper in older Transformers versions. It is not used
    # by VITS, but the package imports XTTS modules at import time.
    import warnings

    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(pytorch_utils, "isin_mps_friendly"):
        pytorch_utils.isin_mps_friendly = torch.isin

    from TTS.tts.configs.shared_configs import CharactersConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coqui_config = VitsConfig()
        coqui_config.load_json(str(config_path))

    characters = dict(raw_config.get("characters", {}))
    if isinstance(characters.get("characters"), list):
        characters["characters"] = "".join(characters["characters"])
    if characters:
        coqui_config.characters = CharactersConfig(**characters)

    model = Vits.init_from_config(coqui_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Coqui VITS checkpoint did not load cleanly: "
            f"missing={load_result.missing_keys[:10]}, unexpected={load_result.unexpected_keys[:10]}"
        )
    checkpoint_step = str(checkpoint.get("step", "?")) if isinstance(checkpoint, dict) else "?"
    return model, checkpoint_step


def synthesize(text: str, output_path: str, model_name: str | None = None) -> None:
    normalized = _normalize_model_name(model_name)
    model, config = get_model(normalized)
    state = _tts_models[normalized]
    processed_text = prepare_text(text)
    print(f"[tts] model={normalized} input: {text}")
    print(f"[tts] model={normalized} mapped: {processed_text}")

    ids = text_to_ids(processed_text)
    text_tensor = torch.IntTensor(ids).unsqueeze(0)
    lengths = torch.IntTensor([len(ids)])
    if state.device == "cuda":
        text_tensor = text_tensor.cuda()
        lengths = lengths.cuda()

    start_time = time.time()
    if hasattr(model, "inference"):
        output = model.inference(text_tensor, {"x_lengths": lengths})
    else:
        output = model.infer(text_tensor, lengths)
    wav = output["model_outputs"].cpu().numpy().flatten()
    sample_rate = config["sample_rate"]

    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    wav_int16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_int16)
    elapsed = time.time() - start_time
    print(f"[tts] model={normalized} wrote: {output_path} ({elapsed:.2f}s, {len(wav)/sample_rate:.2f}s audio)")


if __name__ == "__main__":
    synthesize("shi33 yan55 kua33 chi21", str(ROOT / "shanghai_test.wav"), model_name="shanghai")
