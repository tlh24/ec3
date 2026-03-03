"""Lifter — iterative-inference transformer for LOGO inverse graphics
============================================================================

Task
----
Given two rendered LOGO images A and B, and the program that produced A,
predict the program that produced B.

Inputs:
    x:      [B, 2, 30, 30]  float32  — image A in channel 0, image B in channel 1
    y:      [B, 96]         int64    — y[:, :48] = prog A tokens,
                                       y[:, 48:] = prog B tokens (target)

Sequence layout (full bidirectional attention, N_TOK=546 tokens total):
    [ img_A×225 | img_B×225 | prog_A×48 | prog_B_query×48 ]
    (images downsampled to 15×15 via Conv2d before tokenisation)

Key ideas
---------
* 2D sinusoidal PE (dims 0:32) injected into image token slots before each of
  the first 3 blocks.  Both img_A and img_B get the same 30×30 2D grid.
* Linear sinusoidal PE (dims 0:32) injected into all 96 program token slots
  before each of the last 3 blocks.  Positions 0..47 = prog_A, 48..95 = prog_B.
* Dims 0:32 are the "SPE region"; unembedding reads dims 32:256 only.
* prog_A tokens are embedded via nn.Embedding and supplied as hard input.
* prog_B positions use learned query vectors; loss is computed over these 48 tokens.
* Inference Latent Vectors (ILV): shape [B, N_LAYERS-1, 1896, 256], initialised
  with Gaussian noise each batch, then optimised via manual SGD to minimise the
  cross-entropy loss — WITHOUT touching model weights.  Weights are updated once
  with the final ILV held fixed.
* Optional convolutional amortisation network predicts warm-start ILV from image.

Usage
-----
    model  = Lifter()
    losses = model.train_step(x, y)          # x:[B,2,30,30], y:[B,96] int64
    logits = model.predict(x, y[:, :48])     # [B, 48, 64]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# ============================================================================
# Hyperparameters  — configure here
# ============================================================================

# Transformer
N_LAYERS = 6
N_HEADS  = 4
D_MODEL  = 256
D_FF     = D_MODEL * 4      # FFN hidden dim (nanoGPT: 4x)

# Task geometry
IMG_H = IMG_W = 30          # raw image size
IMG_DS    = 15              # spatial size after Conv2d(1,16,k=4,s=2,p=1) downsampling
N_IMG_ONE = IMG_DS * IMG_DS # 225 tokens per image
N_IMG     = N_IMG_ONE * 2   # 450 image tokens total (img_A + img_B)
N_PROG_A  = 48              # program A tokens (embedded input)
N_PROG_B  = 48              # program B tokens (query / prediction target)
N_PROG    = N_PROG_A + N_PROG_B  # 96
VOCAB     = 64              # token vocabulary size
N_TOK     = N_IMG + N_PROG  # 1896 total sequence length

# SPE: both 2D and linear PEs occupy dims 0:D_SPE of the hidden state.
#   2D PE  (image tokens, first N_LAYERS//2 blocks):
#       8 freqs per axis x (sin+cos) x 2 axes = 32 dims
#   Linear PE (program tokens, last N_LAYERS//2 blocks):
#       16 freqs x (sin+cos) = 32 dims
D_SPE       = 32
N_FREQS_2D  = 8             # per spatial axis; period range [MIN_P_2D, MAX_P_2D]
MIN_P_2D    = 4.0
MAX_P_2D    = 32.0
N_FREQS_LIN = 16            # period range [MIN_P_LIN, MAX_P_LIN]
MIN_P_LIN   = 4.0
MAX_P_LIN   = 128.0

D_CLEAN     = D_MODEL - D_SPE          # 224: SPE-free dims used for unembedding
N_ENC       = N_LAYERS // 2            # 3: blocks that inject 2D PE (image side)

# ILV
N_ILV    = N_LAYERS - 1    # 5: ILV[l] added before block l, for l = 0 .. N_ILV-1
N_INF    = 16               # iterative inference passes per batch
ETA_ILV  = 2e-2            # SGD step size for ILV
WD_ILV   = 0.1             # manual L2 coefficient applied every ILV step
L1_ILV   = 1e-4            # L1 coefficient (only if USE_L1=True)
STD_ILV  = 0.05            # ILV initialisation noise std dev

# Optimisers
LR_W     = 1e-4
WD_W     = 0.01
LR_AMORT = 1e-4

# Amortisation: predict ILV for the first N_ENC transformer layers from image
N_AMORT  = N_ENC            # 3

# Feature flags
USE_L1    = False    # L1 regularisation on ILV during inference passes
USE_AMORT = False    # warm-start ILV from convolutional amortisation network


# ============================================================================
# Sinusoidal PE builders
# ============================================================================

def _geom_periods(n: int, lo: float, hi: float) -> torch.Tensor:
    """n geometrically-spaced periods in [lo, hi]."""
    return lo * (hi / lo) ** torch.linspace(0.0, 1.0, n)


def build_2d_spe() -> torch.Tensor:
    """
    2D sinusoidal PE for a row-major IMG_DS x IMG_DS downsampled image grid.
    Token k = i*IMG_DS + j encodes x=j (column), y=i (row).
    Returns [N_IMG_ONE, D_SPE] = [225, 32].
    Layout: [sin/cos(x*w0)..sin/cos(x*w7) | sin/cos(y*w0)..sin/cos(y*w7)]
    Used for both image A and image B (same spatial grid).
    """
    periods = _geom_periods(N_FREQS_2D, MIN_P_2D, MAX_P_2D)         # [8]
    coords  = torch.arange(IMG_DS, dtype=torch.float32)              # [15]
    ang     = 2 * math.pi * coords[:, None] / periods[None, :]       # [15, 8]
    enc1d   = torch.cat([ang.sin(), ang.cos()], dim=1)               # [15, 16]

    # Row-major token ordering: token k -> col = k % IMG_DS, row = k // IMG_DS
    cols = torch.arange(IMG_DS).repeat(IMG_DS)             # [225] x-coords
    rows = torch.arange(IMG_DS).repeat_interleave(IMG_DS)  # [225] y-coords
    enc_x = enc1d[cols]                                    # [225, 16]
    enc_y = enc1d[rows]                                    # [225, 16]
    return torch.cat([enc_x, enc_y], dim=1)                # [225, 32]


def build_linear_spe() -> torch.Tensor:
    """
    1D sinusoidal PE for N_PROG program tokens.
    Returns [N_PROG, D_SPE] = [96, 32].
    """
    periods = _geom_periods(N_FREQS_LIN, MIN_P_LIN, MAX_P_LIN)  # [16]
    pos     = torch.arange(N_PROG, dtype=torch.float32)           # [96]
    ang     = 2 * math.pi * pos[:, None] / periods[None, :]       # [96, 16]
    return torch.cat([ang.sin(), ang.cos()], dim=1)               # [96, 32]


# ============================================================================
# Transformer building blocks  (nanoGPT style: pre-LN, GELU, no proj bias)
# ============================================================================

class SelfAttention(nn.Module):
    """
    Full bidirectional multi-head self-attention.
    Uses F.scaled_dot_product_attention which dispatches to FlashAttention
    when inputs are on CUDA and no explicit attention mask is given.
    """
    def __init__(self, d: int, nh: int):
        super().__init__()
        assert d % nh == 0
        self.nh   = nh
        self.dh   = d // nh
        self.qkv  = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def rshp(t):
            return t.view(B, T, self.nh, self.dh).transpose(1, 2)
        q, k, v = rshp(q), rshp(k), rshp(v)

        # FlashAttention path (no causal mask -> full bidirectional attention)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )                                                    # [B, nh, T, dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Linear -> GELU -> Linear, 4x hidden expansion."""
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN transformer block: LN->Attn->residual, LN->FFN->residual."""
    def __init__(self, d: int, nh: int, ff: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = SelfAttention(d, nh)
        self.ln2  = nn.LayerNorm(d)
        self.mlp  = MLP(d, ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp( self.ln2(x))
        return x


# ============================================================================
# Convolutional amortisation network
# ============================================================================

class AmortizationNet(nn.Module):
    """
    Hierarchical convnet predicting ILV warm-starts for the first 3
    transformer layers (the 2D-SPE / image-focused layers) from the raw image.

    Architecture (U-Net-style channel doubling, halved channel counts):
      Stage 0 (convs 1-2): Conv(2->16, k=5, s=2), Conv(16->32, k=5, s=2)
                            30x30 -> 8x8 spatial, 32 channels
                            Input has 2 channels: image A and image B stacked.
      Stage 1 (convs 3-4): Conv(32->64, k=3, s=1), Conv(64->64, k=3, s=1)
                            8x8 unchanged, 64 channels
      Stage 2 (convs 5-6): Conv(64->128, k=3, s=1), Conv(128->128, k=3, s=1)
                            8x8 unchanged, 128 channels

    Each stage: GlobalAvgPool -> Linear(ch, n_tok*d) -> reshape [n_tok, d].
    Always produces exactly 3 predictions, one per ILV layer.
    """
    def __init__(self, n_tok: int, d: int):
        super().__init__()
        self.n_tok = n_tok
        self.d     = d

        self.stage0 = nn.Sequential(
            nn.Conv2d( 2, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.GELU(),
        )  # [B, 32, 8, 8]

        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )  # [B, 64, 8, 8]

        self.stage2 = nn.Sequential(
            nn.Conv2d( 64, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
        )  # [B, 128, 8, 8]

        self.heads = nn.ModuleList([
            nn.Linear( 32, n_tok * d),
            nn.Linear( 64, n_tok * d),
            nn.Linear(128, n_tok * d),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, H, W]  (img_A in ch 0, img_B in ch 1)  ->  [B, 3, n_tok, d]"""
        B = x.shape[0]

        h0 = self.stage0(x)                                         # [B, 32,  8, 8]
        h1 = self.stage1(h0)                                        # [B, 64,  8, 8]
        h2 = self.stage2(h1)                                        # [B, 128, 8, 8]

        pred0 = self.heads[0](h0.mean((2, 3))).view(B, self.n_tok, self.d)
        pred1 = self.heads[1](h1.mean((2, 3))).view(B, self.n_tok, self.d)
        pred2 = self.heads[2](h2.mean((2, 3))).view(B, self.n_tok, self.d)

        return torch.stack([pred0, pred1, pred2], dim=1)            # [B, 3, n_tok, d]


# ============================================================================
# Lifter -- main model
# ============================================================================

class Lifter(nn.Module):
    """
    Transformer for LOGO inverse graphics with iterative ILV inference.

    Sequence layout (full bidirectional attention, N_TOK=546 tokens):
        [ img_A×225 | img_B×225 | prog_A×48 | prog_B_query×48 ]

    SPE injection (before each block):
        blocks 0..2 : add spe_enc  (2D PE in dims 0:32 of image token slots;
                                    same 15×15 grid for both img_A and img_B)
        blocks 3..5 : add spe_dec  (linear PE in dims 0:32 of all 96 prog slots;
                                    positions 0..47 = prog_A, 48..95 = prog_B)

    ILV injection (before each block except the last):
        h <- h + ILV[:, l]   for l = 0 ... N_ILV-1

    Unembedding:
        reads h[:, N_IMG+N_PROG_A:, D_SPE:]  -- SPE-free dims of prog_B tokens.

    Training step per batch:
        1.  ILV <- N(0, STD_ILV^2)  [+ amort warm-start if USE_AMORT]
        2.  for n_inference steps:
                logits = forward(x, prog_a, ILV)
                loss   = cross_entropy(logits, prog_b)  [+ L1(ILV) if USE_L1]
                g      = autograd.grad(loss, ILV)  # weights untouched
                ILV   -= eta_ilv * (g + wd_ilv * ILV)
        3.  ILV.detach()
            forward(x, prog_a, ILV); weight_loss.backward() -> AdamW(weights)
        4.  [if USE_AMORT]  MSE(amort_net(x), ILV) -> Adam(amort_net)
    """

    def __init__(
        self,
        n_layers:    int   = N_LAYERS,
        n_heads:     int   = N_HEADS,
        d_model:     int   = D_MODEL,
        d_ff:        int   = D_FF,
        n_inference: int   = N_INF,
        eta_ilv:     float = ETA_ILV,
        wd_ilv:      float = WD_ILV,
        lambda_l1:   float = L1_ILV,
        ilv_std:     float = STD_ILV,
        use_l1:      bool  = USE_L1,
        use_amort:   bool  = USE_AMORT,
    ):
        super().__init__()
        assert d_model >= D_SPE, f"d_model={d_model} must be >= D_SPE={D_SPE}"
        assert n_layers >= 2,    "need at least 2 layers"

        # Store config
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.d_model     = d_model
        self.d_ff        = d_ff
        self.n_inference = n_inference
        self.eta_ilv     = eta_ilv
        self.wd_ilv      = wd_ilv
        self.lambda_l1   = lambda_l1
        self.ilv_std     = ilv_std
        self.use_l1      = use_l1
        self.use_amort   = use_amort

        self.n_ilv   = n_layers - 1
        self.n_enc   = 3
        self.n_amort = 3   # layers predicted by amort
        self.d_clean = d_model - D_SPE               # SPE-free dims for unembedding

        # -- Embeddings -------------------------------------------------------
        # Conv2d(1, 16, k=4, s=2, p=1): 30x30 -> 15x15, shared for img_A & img_B
        self.img_conv   = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.img_proj   = nn.Linear(16, d_model, bias=True)  # 16 conv channels -> d_model
        self.prog_embed = nn.Embedding(VOCAB, d_model)        # for prog_A input tokens
        self.prog_query = nn.Parameter(torch.randn(1, N_PROG_B, d_model) * 0.02)  # prog_B queries

        # -- Transformer ------------------------------------------------------
        self.blocks   = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # -- Unembedding (reads SPE-free dims only) ---------------------------
        self.unembed = nn.Linear(self.d_clean, VOCAB, bias=False)

        # -- SPE buffers  [1, N_TOK, d_model] ---------------------------------
        #   spe_enc: 2D PE in dims 0:D_SPE of image token slots (A and B share
        #            the same 30x30 grid); zeros for program token slots.
        #   spe_dec: linear PE in dims 0:D_SPE of all 96 program token slots
        #            (positions 0..47 = prog_A, 48..95 = prog_B); zeros for images.
        spe2d  = build_2d_spe()      # [900,  D_SPE]  one image worth
        spelin = build_linear_spe()  # [96,   D_SPE]

        enc_buf = torch.zeros(N_TOK, d_model)
        enc_buf[:N_IMG_ONE,            :D_SPE] = spe2d   # img_A
        enc_buf[N_IMG_ONE:N_IMG,       :D_SPE] = spe2d   # img_B (same grid)

        dec_buf = torch.zeros(N_TOK, d_model)
        dec_buf[N_IMG:,                :D_SPE] = spelin  # all 96 prog tokens

        self.register_buffer('spe_enc', enc_buf.unsqueeze(0))  # [1, N_TOK, d]
        self.register_buffer('spe_dec', dec_buf.unsqueeze(0))  # [1, N_TOK, d]

        # -- Optional amortisation --------------------------------------------
        if use_amort:
            self.amort_net = AmortizationNet(N_TOK, d_model)
            self.amort_opt = torch.optim.Adam(
                self.amort_net.parameters(), lr=LR_AMORT)
        else:
            self.amort_net = None
            self.amort_opt = None

        # -- Weight optimiser (excludes amort_net, which has its own) ---------
        amort_ids   = {id(p) for p in (self.amort_net.parameters()
                                       if self.amort_net else [])}
        main_params = [p for p in self.parameters() if id(p) not in amort_ids]
        self.weight_opt = torch.optim.AdamW(
            main_params, lr=LR_W, weight_decay=WD_W)

    # -------------------------------------------------------------------------
    def forward(self,
                x:      torch.Tensor,
                prog_a: torch.Tensor,
                ilv:    Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:      [B, 2, H, W]               img_A in ch 0, img_B in ch 1
            prog_a: [B, N_PROG_A]              int64 program A token indices
            ilv:    [B, N_ILV, N_TOK, d_model] Inference Latent Vector, or None

        Returns:
            logits [B, N_PROG_B, VOCAB]
        """
        B = x.shape[0]

        # Downsample each image: [B,1,30,30] -> [B,16,15,15] -> [B,225,16]
        # then project to d_model; img_conv and img_proj are shared for A and B.
        def embed_img(img):
            f = self.img_conv(img.unsqueeze(1))              # [B, 16, 15, 15]
            f = f.permute(0, 2, 3, 1).reshape(B, N_IMG_ONE, 16)  # [B, 225, 16]
            return self.img_proj(f)                          # [B, 225, d]

        img_a_emb = embed_img(x[:, 0])   # [B, 225, d]
        img_b_emb = embed_img(x[:, 1])   # [B, 225, d]

        # Embed program A (discrete tokens) and program B (learned queries)
        prog_a_emb = self.prog_embed(prog_a.long())                   # [B,  48, d]
        prog_b_emb = self.prog_query.expand(B, -1, -1)               # [B,  48, d]

        h = torch.cat([img_a_emb, img_b_emb, prog_a_emb, prog_b_emb], dim=1)  # [B, N_TOK, d]

        for l, block in enumerate(self.blocks):
            # ILV injection (not on the final block)
            if ilv is not None and l < self.n_ilv:
                h = h + ilv[:, l]

            # SPE injection: 2D for first n_enc blocks, linear for the rest
            h = h + (self.spe_enc if l < self.n_enc else self.spe_dec)

            h = block(h)

        h = self.ln_final(h)                                          # [B, 1896, d]

        # Unembedding: prog_B token slots only, SPE-free dims
        prog_b_h = h[:, N_IMG + N_PROG_A:, D_SPE:]                   # [B, 48, d_clean]
        return self.unembed(prog_b_h)                                 # [B, 48, VOCAB]

    # -------------------------------------------------------------------------
    def _init_ilv(self, B: int, device: torch.device,
                  x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Allocate a fresh ILV leaf tensor [B, N_ILV, N_TOK, d_model].
        Initialised with Gaussian noise.  If use_amort and x is provided,
        the amortisation network's output is added to the first n_amort layers
        before requires_grad is set.
        """
        ilv = torch.randn(
            B, self.n_ilv, N_TOK, self.d_model, device=device
        ) * self.ilv_std

        if self.use_amort and x is not None and self.amort_net is not None:
            with torch.no_grad():
                ilv[:, :self.n_amort].add_(self.amort_net(x))

        return ilv.requires_grad_(True)

    # -------------------------------------------------------------------------
    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor) -> Dict[str, float]:
        """
        One full training step for a single batch.

        Args:
            x: [B, 2, H, W]   float32, img_A in ch 0, img_B in ch 1
            y: [B, 96]         int64,   y[:, :48] = prog_A, y[:, 48:] = prog_B

        Returns dict with keys:
            'ilv_loss'    -- CE loss at the end of the ILV inference loop
            'weight_loss' -- CE loss used for the weight gradient update
            'amort_loss'  -- MSE loss for amortisation net (0.0 if disabled)
        """
        self.train()
        B, dev = x.shape[0], x.device

        prog_a = y[:, :N_PROG_A].long()   # [B, 48] — supplied as input
        prog_b = y[:, N_PROG_A:].long()   # [B, 48] — prediction target

        # 1. Initialise ILV ---------------------------------------------------
        ilv = self._init_ilv(B, dev, x)

        # 2. Iterative ILV optimisation (weights stay frozen) -----------------
        ilv_loss_sta = 0.0
        ilv_loss_fin = 0.0
        for u in range(self.n_inference):
            logits = self.forward(x, prog_a, ilv)
            loss   = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))
            if self.use_l1:
                loss = loss + self.lambda_l1 * ilv.abs().sum()

            # Gradient only w.r.t. ILV -- avoids accumulating weight grads
            (g,) = torch.autograd.grad(loss, ilv)

            # Manual SGD update with L2 weight decay
            with torch.no_grad():
                ilv.data -= self.eta_ilv * (g + self.wd_ilv * ilv.data)

            if u == 0: 
                ilv_loss_sta = loss.item()
            if u == self.n_inference-1:
                ilv_loss_fin = loss.item()

        # 3. Weight update with ILV held fixed --------------------------------
        ilv_fixed = ilv.detach()
        self.weight_opt.zero_grad()
        logits = self.forward(x, prog_a, ilv_fixed)
        w_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), prog_b.reshape(-1)
        )
        w_loss.backward()
        self.weight_opt.step()

        # 4. Amortisation update ----------------------------------------------
        amort_loss_v = 0.0
        if self.use_amort and self.amort_opt is not None:
            # Train amort_net to predict the optimised ILV from raw pixels
            target = ilv_fixed[:, :self.n_amort]    # [B, n_amort, N_TOK, d]
            self.amort_opt.zero_grad()
            a_loss = F.mse_loss(self.amort_net(x), target)
            a_loss.backward()
            self.amort_opt.step()
            amort_loss_v = a_loss.item()

        return logits, {
            'ilv_loss':    ilv_loss_sta - ilv_loss_fin,
            # reports how much the loss changes during ILV optimization
            'weight_loss': w_loss.item(),
            'amort_loss':  amort_loss_v,
        }

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, prog_a: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass -- no ILV gradient optimisation.
        If amortisation is enabled the ILV is warm-started from the convnet;
        otherwise ILV=None (no injection).

        Args:
            x:      [B, 2, H, W]   img_A in ch 0, img_B in ch 1
            prog_a: [B, N_PROG_A]  int64 program A tokens

        Returns logits [B, N_PROG_B, VOCAB].
        """
        self.eval()
        if self.use_amort and self.amort_net is not None:
            B, dev = x.shape[0], x.device
            ilv = torch.randn(
                B, self.n_ilv, N_TOK, self.d_model, device=dev
            ) * self.ilv_std
            ilv[:, :self.n_amort].add_(self.amort_net(x))
            return self.forward(x, prog_a.long(), ilv)
        return self.forward(x, prog_a.long(), ilv=None)

    # -------------------------------------------------------------------------
    def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
        """Save model weights and optimiser states to a checkpoint file."""
        checkpoint = {
            'model_state':      self.state_dict(),
            'weight_opt_state': self.weight_opt.state_dict(),
            'step':             step,
        }
        if self.use_amort and self.amort_opt is not None:
            checkpoint['amort_opt_state'] = self.amort_opt.state_dict()
        torch.save(checkpoint, path)

    # -------------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> Optional[int]:
        """
        Load model weights and optimiser states from a checkpoint file.
        Returns the saved step (or None if not present).
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['model_state'])
        self.weight_opt.load_state_dict(checkpoint['weight_opt_state'])
        if self.use_amort and self.amort_opt is not None and 'amort_opt_state' in checkpoint:
            self.amort_opt.load_state_dict(checkpoint['amort_opt_state'])
        return checkpoint.get('step')

    # -------------------------------------------------------------------------
    def flops_per_forward(self) -> int:
        """
        Theoretical FLOPs for one forward pass on one sample.
        Multiply-adds count as 2 FLOPs each.
        LayerNorm, activation, and softmax costs are omitted (minor relative
        to matmuls).
        """
        d  = self.d_model
        ff = self.d_ff
        nh = self.n_heads
        dh = d // nh
        T  = N_TOK                               # 546

        # Conv2d(1,16,k=4,s=2): 2 * C_in*C_out*k*k * H_out*W_out, times 2 images
        f  = 2 * 2 * 1 * 16 * 4 * 4 * IMG_DS * IMG_DS
        # img_proj: Linear(16->d) over N_IMG tokens
        f += 2 * N_IMG * 16 * d
        # prog_A embedding is a lookup (no matmul FLOPs counted)

        block_f = (
            2 * T * d * 3 * d +                  # QKV projection
            2 * nh * T * T * dh +                # Q*K^T attention scores
            2 * nh * T * T * dh +                # weighted value aggregation
            2 * T * d * d       +                # output projection
            2 * T * d * ff      +                # FFN fc1
            2 * T * ff * d                       # FFN fc2
        )
        f += self.n_layers * block_f

        f += 2 * N_PROG_B * self.d_clean * VOCAB  # unembedding (prog_B only)

        return f

    # -------------------------------------------------------------------------
    def count_params(self) -> Dict[str, int]:
        """
        Returns a dict:
            weight_params   -- learnable transformer + embedding parameters
            ilv_per_sample  -- ILV elements per batch sample (re-init every batch)
            amort_params    -- amortisation network parameters (0 if disabled)
        """
        amort_ids = {id(p) for p in (self.amort_net.parameters()
                                     if self.amort_net else [])}
        w = sum(p.numel() for p in self.parameters() if id(p) not in amort_ids)
        a = sum(p.numel() for p in (self.amort_net.parameters()
                                    if self.amort_net else []))
        return {
            'weight_params':  w,
            'ilv_per_sample': self.n_ilv * N_TOK * self.d_model,
            'amort_params':   a,
        }


# ============================================================================
# ILV independence diagnostic
# ============================================================================

def test_ilv_independence(model: 'Lifter', device: torch.device,
                          batch_size: int = 2) -> None:
    """
    Verify that:
      1. ILV gradients are non-zero  (autograd graph is connected).
      2. ILV actually changes after each manual SGD step.
      3. ilv_loss and weight_loss in train_step() are not bit-for-bit identical.
      4. Loss trajectory across inference passes is printed for inspection.

    Intended as a quick sanity check when ilv_loss == weight_loss is observed.
    Raises AssertionError on critical failures; prints INFO/WARN for softer issues.
    """
    model.train()
    torch.manual_seed(0)
    x      = torch.rand(batch_size, 2, IMG_H, IMG_W, device=device)
    y      = torch.randint(0, VOCAB, (batch_size, N_PROG), device=device)
    prog_a = y[:, :N_PROG_A].long()
    prog_b = y[:, N_PROG_A:].long()

    print("\n=== test_ilv_independence ===")

    # ---- 1. Gradient connectivity ----------------------------------------
    ilv = model._init_ilv(batch_size, device, x)
    print(f"  ILV shape:           {tuple(ilv.shape)}")
    print(f"  ILV requires_grad:   {ilv.requires_grad}")
    print(f"  ILV data norm (init):{ilv.detach().norm().item():.6f}")

    logits = model.forward(x, prog_a, ilv)
    loss0  = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))
    (g,)   = torch.autograd.grad(loss0, ilv)

    grad_norm = g.norm().item()
    print(f"\n  [1] ILV gradient norm after pass-0: {grad_norm:.6e}")
    assert grad_norm > 1e-8, (
        f"FAIL: ILV gradient is effectively zero (norm={grad_norm:.2e}).\n"
        "      Autograd graph may be disconnected between loss and ILV."
    )
    print(f"      => PASS: gradient is non-zero")

    # ---- 2. ILV data actually changes ------------------------------------
    ilv_before = ilv.detach().clone()
    with torch.no_grad():
        ilv.data -= model.eta_ilv * (g + model.wd_ilv * ilv.data)
    delta = (ilv.detach() - ilv_before).norm().item()

    print(f"\n  [2] ILV delta norm after 1 SGD step: {delta:.6e}")
    assert delta > 1e-12, (
        f"FAIL: ILV did not change after SGD step (delta={delta:.2e}).\n"
        "      Check eta_ilv and that ilv.data is being updated in-place."
    )
    print(f"      => PASS: ILV changed")

    # ---- 3. Loss trajectory over all inference passes --------------------
    # Re-init so we start fresh for the trajectory test
    ilv2     = model._init_ilv(batch_size, device, x)
    losses_v = []
    for step in range(model.n_inference):
        logits_i = model.forward(x, prog_a, ilv2)
        loss_i   = F.cross_entropy(logits_i.reshape(-1, VOCAB), prog_b.reshape(-1))
        losses_v.append(loss_i.item())
        (g_i,)   = torch.autograd.grad(loss_i, ilv2)
        with torch.no_grad():
            ilv2.data -= model.eta_ilv * (g_i + model.wd_ilv * ilv2.data)

    print(f"\n  [3] Loss per inference pass (n_inference={model.n_inference}):")
    for i, lv in enumerate(losses_v):
        marker = " <-- ilv_loss" if i == len(losses_v) - 1 else ""
        print(f"      pass {i:2d}: {lv:.10f}{marker}")

    # Weight-update forward with fixed ILV
    ilv_fixed = ilv2.detach()
    w_logits  = model.forward(x, prog_a, ilv_fixed)
    w_loss_v  = F.cross_entropy(
        w_logits.reshape(-1, VOCAB), prog_b.reshape(-1)
    ).item()
    print(f"      weight :  {w_loss_v:.10f}  <-- weight_loss")

    ilv_loss_v = losses_v[-1]
    diff       = abs(ilv_loss_v - w_loss_v)
    print(f"\n      |ilv_loss - weight_loss| = {diff:.3e}")

    if ilv_loss_v == w_loss_v:
        # Bit-for-bit identical: indicates the ILV update had zero effect
        print("      WARNING: losses are bit-for-bit identical!")
        print("               Likely cause: gradient magnitude is negligible,")
        print("               so ILV barely moves and both passes see same weights+ILV.")
        print(f"               grad_norm={grad_norm:.2e}, eta_ilv={model.eta_ilv}")
    else:
        print(f"      => PASS: ilv_loss != weight_loss (losses are independent)")

    # ---- 4. train_step round-trip (high-precision print) -----------------
    losses_ts = model.train_step(x, y)
    print(f"\n  [4] train_step() round-trip (high precision):")
    print(f"      ilv_loss    = {losses_ts['ilv_loss']:.10f}")
    print(f"      weight_loss = {losses_ts['weight_loss']:.10f}")
    diff_ts = abs(losses_ts['ilv_loss'] - losses_ts['weight_loss'])
    print(f"      difference  = {diff_ts:.3e}")
    if losses_ts['ilv_loss'] == losses_ts['weight_loss']:
        print("      WARNING: train_step losses are bit-for-bit identical!")
    else:
        print("      => PASS: train_step losses differ")


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}\n")

    # -- Without amortisation -------------------------------------------------
    print("=== No amortisation ===")
    model = Lifter(use_amort=False).to(dev)
    p = model.count_params()
    print(f"  Weight params:     {p['weight_params']:>12,}")
    print(f"  ILV / sample:      {p['ilv_per_sample']:>12,}")
    print(f"  Amort params:      {p['amort_params']:>12,}")
    gf = model.flops_per_forward() / 1e9
    print(f"  FLOPs/fwd/sample:  {gf:.3f} GFLOPs")

    B = 4
    x = torch.rand(B, 2, IMG_H, IMG_W, device=dev)       # two images per sample
    y = torch.randint(0, VOCAB, (B, N_PROG), device=dev)  # prog_A in [:48], prog_B in [48:]
    prog_a = y[:, :N_PROG_A]

    losses = model.train_step(x, y)
    print(f"  ILV loss:          {losses['ilv_loss']:.4f}")
    print(f"  Weight loss:       {losses['weight_loss']:.4f}")

    logits = model.predict(x, prog_a)
    assert logits.shape == (B, N_PROG_B, VOCAB), f"Bad shape: {logits.shape}"
    print(f"  predict() shape:   {tuple(logits.shape)}  OK")

    # -- With amortisation ----------------------------------------------------
    print("\n=== With amortisation ===")
    model2 = Lifter(use_amort=True).to(dev)
    p2 = model2.count_params()
    print(f"  Weight params:     {p2['weight_params']:>12,}")
    print(f"  ILV / sample:      {p2['ilv_per_sample']:>12,}")
    print(f"  Amort params:      {p2['amort_params']:>12,}")

    losses2 = model2.train_step(x, y)
    print(f"  ILV loss:          {losses2['ilv_loss']:.4f}")
    print(f"  Weight loss:       {losses2['weight_loss']:.4f}")
    print(f"  Amort loss:        {losses2['amort_loss']:.6f}")

    logits2 = model2.predict(x, prog_a)
    assert logits2.shape == (B, N_PROG_B, VOCAB)
    print(f"  predict() shape:   {tuple(logits2.shape)}  OK")

    # -- L1 flag check --------------------------------------------------------
    print("\n=== USE_L1 flag ===")
    model3 = Lifter(use_l1=True, n_inference=2).to(dev)
    losses3 = model3.train_step(x, y)
    print(f"  ILV loss (L1 on):  {losses3['ilv_loss']:.4f}  OK")

    # -- ILV independence diagnostic (catches ilv_loss == weight_loss bugs) --
    test_ilv_independence(model, dev)
