"""Deep models (PyTorch, CPU-friendly; everything degrades gracefully when torch is missing).

1) MaskedClimateTransformer — BERT-style masked modelling of multivariate daily weather windows.
   • days are tokens; 15–35 % of days are masked (80 % [MASK] token, 10 % random day, 10 % kept — BERT recipe)
   • reconstruction error on masked days = "the atmosphere did something the model cannot infer from context"
   • LoRA ADAPTER INJECTION: every attention/FFN projection is a LoRALinear. The base model learns the
     climate of the baseline decades; a low-rank adapter is then injected and trained only on the
     recent period. ‖ΔW‖ and the error reduction it buys measure REGIME DRIFT ("how much the climate
     had to be re-learnt"). Adapters are stored in a bank keyed by place/regime and can be re-injected
     for similar places retrieved from memory.
   • MASK-GAN: a 1-D convolutional critic tries to tell real windows from windows whose masked days were
     filled by the generator (the transformer). Generator loss = masked MSE + λ·adversarial. At test time
     the critic's 'fake-ness' of the REAL window is an extra anomaly signal (GANomaly / TadGAN idea).

2) CompassMaskedImageGAN — masked-patch inpainting of satellite tiles with an encoder built from the
   QDiRP-CompassNet directional depthwise mixer + Compass (axial/diagonal) aggregation (after S. Ghandali's
   technical note). A PatchGAN critic judges realism. Trained on the
   same calendar window in previous years; today's inpainting error vs that distribution → image anomaly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except Exception:  # noqa: BLE001
    TORCH = False


def torch_available() -> bool:
    return TORCH


if TORCH:
    # ------------------------------------------------------------------------------------- LoRA
    class LoRALinear(nn.Module):
        def __init__(self, d_in: int, d_out: int, rank: int = 4, alpha: float = 8.0) -> None:
            super().__init__()
            self.base = nn.Linear(d_in, d_out)
            self.rank, self.scale = rank, alpha / rank
            self.A: nn.Parameter | None = None
            self.B: nn.Parameter | None = None

        def inject(self) -> None:
            dev = self.base.weight.device
            self.A = nn.Parameter(torch.randn(self.rank, self.base.in_features, device=dev) * 0.01)
            self.B = nn.Parameter(torch.zeros(self.base.out_features, self.rank, device=dev))

        def remove(self) -> None:
            self.A, self.B = None, None

        def delta_norm(self) -> float:
            if self.A is None:
                return 0.0
            return float((self.B @ self.A).norm() * self.scale / (self.base.weight.norm() + 1e-9))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.base(x)
            if self.A is not None:
                y = y + (x @ self.A.t() @ self.B.t()) * self.scale
            return y

    class Block(nn.Module):
        def __init__(self, d: int, heads: int, rank: int) -> None:
            super().__init__()
            self.h, self.dk = heads, d // heads
            self.qkv = LoRALinear(d, 3 * d, rank)
            self.proj = LoRALinear(d, d, rank)
            self.ff1 = LoRALinear(d, 2 * d, rank)
            self.ff2 = LoRALinear(2 * d, d, rank)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.drop = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
            q = q.view(B, L, self.h, self.dk).transpose(1, 2)
            k = k.view(B, L, self.h, self.dk).transpose(1, 2)
            v = v.view(B, L, self.h, self.dk).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
            o = (att @ v).transpose(1, 2).reshape(B, L, D)
            x = x + self.drop(self.proj(o))
            x = x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))
            return x

    class MaskedClimateTransformer(nn.Module):
        def __init__(self, n_feat: int, d: int = 64, layers: int = 3, heads: int = 4, max_len: int = 64, rank: int = 4) -> None:
            super().__init__()
            self.inp = nn.Linear(n_feat + 2, d)            # + sin/cos day-of-year
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
            self.blocks = nn.ModuleList([Block(d, heads, rank) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, n_feat)

        def lora_layers(self):
            return [m for m in self.modules() if isinstance(m, LoRALinear)]

        def forward(self, x: torch.Tensor, doy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.inp(torch.cat([x, torch.sin(doy)[..., None], torch.cos(doy)[..., None]], dim=-1))
            h = torch.where(mask[..., None], self.mask_token.expand_as(h), h)
            h = h + self.pos[:, : h.shape[1]]
            for b in self.blocks:
                h = b(h)
            return self.out(self.norm(h))

    class Critic1D(nn.Module):
        def __init__(self, n_feat: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(n_feat, 32, 5, padding=2), nn.LeakyReLU(0.2),
                                     nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
                                     nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, L, F]
            return self.net(x.transpose(1, 2)).squeeze(-1)

    # ------------------------------------------------------------------------------------- vision (QDiRP-Compass)
    class DirectionalMixer(nn.Module):
        """Five depthwise paths (k×k, 1×k, k×1, 1×1, identity) + shared BN — re-parameterisable (QDiRP)."""

        def __init__(self, ch: int, k: int = 3) -> None:
            super().__init__()
            p = k // 2
            self.sq = nn.Conv2d(ch, ch, k, padding=p, groups=ch, bias=False)
            self.h = nn.Conv2d(ch, ch, (1, k), padding=(0, p), groups=ch, bias=False)
            self.v = nn.Conv2d(ch, ch, (k, 1), padding=(p, 0), groups=ch, bias=False)
            self.pt = nn.Conv2d(ch, ch, 1, groups=ch, bias=False)
            self.bn = nn.BatchNorm2d(ch)

        def forward(self, x):
            return self.bn(self.sq(x) + self.h(x) + self.v(x) + self.pt(x) + x)

    class Compass(nn.Module):
        """Fixed axial (+) and diagonal (×) neighbour averages with learned 1×1 projections — a natural prior
        for wind-streaked and frontal cloud bands."""

        def __init__(self, ch: int) -> None:
            super().__init__()
            ax = torch.zeros(ch, 1, 3, 3)
            ax[:, 0, 0, 1] = ax[:, 0, 2, 1] = ax[:, 0, 1, 0] = ax[:, 0, 1, 2] = 0.25
            dg = torch.zeros(ch, 1, 3, 3)
            dg[:, 0, 0, 0] = dg[:, 0, 0, 2] = dg[:, 0, 2, 0] = dg[:, 0, 2, 2] = 0.25
            self.register_buffer("ax", ax)
            self.register_buffer("dg", dg)
            self.pa = nn.Conv2d(ch, ch, 1, bias=False)
            self.pd = nn.Conv2d(ch, ch, 1, bias=False)
            self.bn = nn.BatchNorm2d(ch)
            self.ch = ch

        def forward(self, x):
            a = F.conv2d(x, self.ax, padding=1, groups=self.ch)
            d = F.conv2d(x, self.dg, padding=1, groups=self.ch)
            return F.relu(x + self.bn(self.pa(a) + self.pd(d)))

    class CompassEncoder(nn.Module):
        def __init__(self, ch: int = 32) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(4, ch, 3, stride=2, padding=1), nn.BatchNorm2d(ch), nn.ReLU())   # RGB + mask
            self.b1 = nn.Sequential(DirectionalMixer(ch, 3), nn.ReLU(), nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1), nn.ReLU())
            self.b2 = nn.Sequential(DirectionalMixer(2 * ch, 5), nn.ReLU(), Compass(2 * ch))

        def forward(self, x):
            return self.b2(self.b1(self.stem(x)))

    class ImageInpainter(nn.Module):
        def __init__(self, ch: int = 32) -> None:
            super().__init__()
            self.enc = CompassEncoder(ch)
            self.dec = nn.Sequential(nn.ConvTranspose2d(2 * ch, ch, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(ch, 16, 4, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

        def forward(self, img, mask):  # img [B,3,H,W] in [0,1]; mask [B,1,H,W] 1 = hidden
            x = torch.cat([img * (1 - mask), mask], dim=1)
            return self.dec(self.enc(x))

    class PatchCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(32, 64, 4, 2, 1),
                                     nn.LeakyReLU(0.2), nn.Conv2d(64, 1, 3, 1, 1))

        def forward(self, x):
            return self.net(x)


# ========================================================================================= wrappers
@dataclass
class DeepResult:
    ok: bool
    note: str = ""
    day_scores: np.ndarray | None = None          # [n_target_days] in [0,1]
    per_var_error: np.ndarray | None = None       # [n_target_days, F]
    critic_fakeness: np.ndarray | None = None
    drift: dict = field(default_factory=dict)
    loss_curve: list = field(default_factory=list)


def _windows(x: np.ndarray, L: int, stride: int) -> np.ndarray:
    idx = [i for i in range(0, len(x) - L + 1, stride)]
    return np.stack([x[i:i + L] for i in idx]) if idx else np.zeros((0, L, x.shape[1]))


def _bert_mask(B: int, L: int, ratio: float, gen) -> "torch.Tensor":
    m = torch.rand(B, L, generator=gen) < ratio
    return m


def run_masked_transformer(base: np.ndarray, base_doy: np.ndarray, target: np.ndarray, target_doy: np.ndarray,
                           recent: np.ndarray | None = None, recent_doy: np.ndarray | None = None,
                           epochs: int = 6, L: int = 32, seed: int = 0, adv_weight: float = 0.05) -> DeepResult:
    """base/target/recent: standardised [T, F] arrays (NaN-free); *_doy: day-of-year in RADIANS (2π·doy/365.25).
    Returns per-target-day anomaly scores."""
    if not TORCH:
        return DeepResult(False, "PyTorch not installed — using the masked linear reconstructor instead")
    try:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        Fn = base.shape[1]
        model = MaskedClimateTransformer(Fn, max_len=L)
        critic = Critic1D(Fn)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.5, 0.9))
        wb = torch.tensor(_windows(base, L, 4), dtype=torch.float32)
        db = torch.tensor(_windows(base_doy[:, None], L, 4)[..., 0], dtype=torch.float32)
        curve = []

        def train(Xw, Dw, n_epochs, params_opt, train_critic=True):
            nonlocal curve
            n = Xw.shape[0]
            for _ in range(n_epochs):
                perm = torch.randperm(n, generator=gen)
                tot = 0.0
                for s in range(0, n, 64):
                    xb, dbb = Xw[perm[s:s + 64]], Dw[perm[s:s + 64]]
                    B = xb.shape[0]
                    mask = _bert_mask(B, L, 0.25, gen)
                    if not bool(mask.any()):
                        continue
                    # BERT 80/10/10: of the masked positions, 10 % get a random day, 10 % are kept visible
                    r = torch.rand(B, L, generator=gen)
                    rand_pos = mask & (r < 0.1)
                    keep_pos = mask & (r >= 0.1) & (r < 0.2)
                    x_in = xb.clone()
                    if rand_pos.any():
                        x_in[rand_pos] = xb.reshape(-1, Fn)[torch.randint(0, B * L, (int(rand_pos.sum()),), generator=gen)]
                    token_mask = mask & ~rand_pos & ~keep_pos
                    pred = model(x_in, dbb, token_mask)
                    mse = ((pred - xb) ** 2)[mask].mean()
                    filled = torch.where(mask[..., None], pred, xb)
                    loss = mse
                    if train_critic and adv_weight > 0:
                        # critic step (non-saturating GAN)
                        c_real = critic(xb)
                        c_fake = critic(filled.detach())
                        lc = F.softplus(-c_real).mean() + F.softplus(c_fake).mean()
                        opt_c.zero_grad()
                        lc.backward()
                        opt_c.step()
                        loss = loss + adv_weight * F.softplus(-critic(filled)).mean()
                    params_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    params_opt.step()
                    tot += float(mse) * B
                curve.append(tot / max(n, 1))

        model.train()
        train(wb, db, epochs, opt)

        @torch.no_grad()
        def score(X: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Mask each day in turn (in a sliding context window) → per-day reconstruction error."""
            model.eval()
            critic.eval()
            T = len(X)
            err = np.zeros((T, Fn))
            cnt = np.zeros(T)
            fake = np.zeros(T)
            Xt = torch.tensor(X, dtype=torch.float32)
            Dt = torch.tensor(D, dtype=torch.float32)
            starts = list(range(0, max(T - L, 0) + 1, max(1, L // 4)))
            if T > L and starts[-1] != T - L:
                starts.append(T - L)
            for s in starts:
                xw = Xt[s:s + L][None]
                dw = Dt[s:s + L][None]
                Lw = xw.shape[1]
                stripes = 4
                for off in range(stripes):
                    mask = torch.zeros(1, Lw, dtype=torch.bool)
                    mask[0, off::stripes] = True
                    pred = model(xw, dw, mask)
                    e = ((pred - xw) ** 2)[0][mask[0]].numpy()
                    pos = np.arange(s, s + Lw)[mask[0].numpy()]
                    err[pos] += e
                    cnt[pos] += 1
                c = float(torch.sigmoid(-critic(xw))[0])
                fake[s:s + Lw] = np.maximum(fake[s:s + Lw], c)
            err = err / np.maximum(cnt, 1)[:, None]
            return err, np.sqrt(err.mean(axis=1)), fake

        base_err, base_rmse, base_fake = score(base[-min(len(base), 1500):], base_doy[-min(len(base), 1500):])
        tgt_err, tgt_rmse, tgt_fake = score(target, target_doy)
        q50, q995 = np.quantile(base_rmse, 0.5), np.quantile(base_rmse, 0.995)
        day_scores = np.clip((tgt_rmse - q50) / (q995 - q50 + 1e-9), 0, 1.5) / 1.5
        per_var = np.sqrt(tgt_err) / (np.sqrt(base_err).std(axis=0) + 1e-6)
        drift = {}
        if recent is not None and len(recent) > L + 8:
            # ---- LoRA adapter injection on the recent regime
            before = float(np.mean(score(recent, recent_doy)[1]))
            for p in model.parameters():
                p.requires_grad_(False)
            for layer in model.lora_layers():
                layer.inject()
            lora_params = [p for layer in model.lora_layers() for p in (layer.A, layer.B)]
            opt_l = torch.optim.Adam(lora_params, lr=3e-3)
            wr = torch.tensor(_windows(recent, L, 2), dtype=torch.float32)
            dr = torch.tensor(_windows(recent_doy[:, None], L, 2)[..., 0], dtype=torch.float32)
            model.train()
            train(wr, dr, max(2, epochs // 2), opt_l, train_critic=False)
            after = float(np.mean(score(recent, recent_doy)[1]))
            dn = [layer.delta_norm() for layer in model.lora_layers()]
            drift = {"recent_rmse_before_adapter": round(before, 4), "recent_rmse_after_adapter": round(after, 4),
                     "error_reduction_pct": round(100 * (before - after) / max(before, 1e-9), 2),
                     "adapter_relative_norm": round(float(np.mean(dn)), 5),
                     "adapter_params": int(sum(p.numel() for p in lora_params))}
        return DeepResult(True, "masked transformer + MaskGAN critic + LoRA adapters", day_scores, per_var, tgt_fake, drift, curve)
    except Exception as exc:  # noqa: BLE001
        return DeepResult(False, f"deep model failed ({type(exc).__name__}: {str(exc)[:160]}) — fallback used")


def run_image_inpainting(base_imgs: list[np.ndarray], target_imgs: list[np.ndarray], epochs: int = 30, seed: int = 0) -> DeepResult:
    """Images: uint8 HxWx3 (H, W divisible by 4). Returns per-target-image anomaly score in [0,1]."""
    if not TORCH:
        return DeepResult(False, "PyTorch not installed — image anomaly uses physical features only")
    if len(base_imgs) < 2 or not target_imgs:
        return DeepResult(False, "not enough baseline images")
    try:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        to_t = lambda arr: torch.tensor(np.stack(arr), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # noqa: E731
        Xb, Xt = to_t(base_imgs), to_t(target_imgs)
        H, W = Xb.shape[-2:]
        G, D = ImageInpainter(), PatchCritic()
        og = torch.optim.Adam(G.parameters(), lr=2e-3)
        od = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

        def patch_mask(B, ratio=0.4, p=16):
            m = (torch.rand(B, 1, H // p, W // p, generator=gen) < ratio).float()
            return F.interpolate(m, size=(H, W), mode="nearest")

        G.train()
        curve = []
        for _ in range(epochs):
            xb = Xb[torch.randint(0, len(Xb), (min(8, len(Xb)),), generator=gen)]
            if torch.rand(1, generator=gen) < 0.5:
                xb = torch.flip(xb, dims=[3])
            m = patch_mask(xb.shape[0])
            out = G(xb, m)
            comp = xb * (1 - m) + out * m
            ld = F.softplus(-D(xb)).mean() + F.softplus(D(comp.detach())).mean()
            od.zero_grad()
            ld.backward()
            od.step()
            rec = (((out - xb) ** 2) * m).sum() / (m.sum() * 3 + 1e-6)
            lg = rec + 0.01 * F.softplus(-D(comp)).mean()
            og.zero_grad()
            lg.backward()
            og.step()
            curve.append(float(rec))

        @torch.no_grad()
        def err(X):
            G.eval()
            es = []
            for i in range(len(X)):
                x = X[i:i + 1]
                tot = 0.0
                for k in range(4):
                    m = patch_mask(1, 0.5)
                    out = G(x, m)
                    tot += float((((out - x) ** 2) * m).sum() / (m.sum() * 3 + 1e-6))
                es.append(tot / 4)
            return np.array(es)

        eb, et = err(Xb), err(Xt)
        z = (et - eb.mean()) / (eb.std() + 1e-6)
        return DeepResult(True, "Compass masked-inpainting GAN", np.clip(z / 4, 0, 1), None, None, {"base_err_mean": float(eb.mean())}, curve)
    except Exception as exc:  # noqa: BLE001
        return DeepResult(False, f"image model failed ({type(exc).__name__}: {str(exc)[:160]})")
