"""InverseGraphics + ForwardGraphics — inverse and forward graphics transformers for LOGO
============================================================================

ForwardGraphics  (prog → 32-channel latent image)
--------------------------------------------------
Given a program (pre-embedded token vectors), predict a 32-channel latent image.
Channel 0 is the grayscale reconstruction; channels 1-31 carry latent information
for the inverse model.

	Sequence layout: [ img_query×225 | prog×48 ]   N_FWD_TOK = 273

	SPE injection (mirror of InverseGraphics):
		blocks 0..2 : linear PE in prog token slots (dims 0:32)
		blocks 3..5 : 2D PE in image token slots   (dims 0:32)

	Unembedding: image token SPE-free dims → Linear → 32ch → ConvTranspose → 30×30

InverseGraphics  (inverse graphics edit: img_A + img_B + prog_A → prog_B)
-----------------------------------------------------------------
Both images are 32-channel: channel 0 = grayscale, channels 1-31 = latent
information from ForwardGraphics (or zeros on the first pass).

	Sequence layout: [ img_A×225 | img_B×225 | prog_A×48 | prog_B_query×48 ]

	SPE injection (unchanged):
		blocks 0..2 : 2D PE in image token slots
		blocks 3..5 : linear PE in program token slots

	Unembedding: prog_B token SPE-free dims → logits [B, 48, VOCAB]

Combined iterative inference (CombinedModel class):
	1. img_a_32 = cat([x[:,0:1], ForwardGraphics(embed_hard(prog_a))[:,1:]], dim=1)
	2. img_b_32 = cat([x[:,1:2], zeros(B,31,H,W)], dim=1)
	3. for k in range(K):
		   logits	  = Lifter(img_a_32, img_b_32, prog_a)
		   soft_emb	= embed_soft(logits)		   # softmax → embedding
		   img_b_fwd   = ForwardGraphics(soft_emb)
		   img_b_32	= cat([x[:,1:2], img_b_fwd[:,1:]], dim=1)
	4. loss = CE(logits, prog_b_gt)
			 + λ * MSE(ForwardGraphics(embed_hard(prog_b_gt))[:,0], x[:,1])
	   loss.backward()   # updates both models jointly
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import pdb

# ============================================================================
# Hyperparameters  — configure here
# ============================================================================

# Transformer
N_LAYERS = 6
N_HEADS  = 4
D_MODEL  = 128
D_FF	 = D_MODEL * 4

# Task geometry
IMG_H = IMG_W = 30
IMG_DS	= 15
N_IMG_ONE = IMG_DS * IMG_DS   # 225 tokens per image
N_IMG	 = N_IMG_ONE * 2	 # 450 image tokens total
N_PROG_A  = 48
N_PROG_B  = 48
N_PROG	= N_PROG_A + N_PROG_B   # 96
VOCAB	 = 64
N_TOK	 = N_IMG + N_PROG		# 546  (Lifter sequence length)

# 32-channel latent image (channel 0 = grayscale, 1-31 = ForwardGraphics latent)
N_IMG_CH  = 32
N_FWD_TOK = N_IMG_ONE + N_PROG_A  # 273  (ForwardGraphics sequence length)

# SPE
D_SPE	   = 32
N_FREQS_2D  = 8
MIN_P_2D	= 4.0
MAX_P_2D	= 32.0
N_FREQS_LIN = 16
MIN_P_LIN   = 4.0
MAX_P_LIN   = 128.0

D_CLEAN = D_MODEL - D_SPE   # 224: SPE-free dims for unembedding
N_ENC   = N_LAYERS // 2	 # 3

# Optimisers
LR_W   = 1e-5
WD_W   = 0.01
LR_FWD = 1e-3
WD_FWD = 0.01

# Combined training
K_INF		= 2	 # forward-inverse iteration passes
LAMBDA_RECON = 10.0   # weight of ForwardGraphics L2 pixel reconstruction loss


# ============================================================================
# Sinusoidal PE builders
# ============================================================================

def _geom_periods(n: int, lo: float, hi: float) -> torch.Tensor:
	"""n geometrically-spaced periods in [lo, hi]."""
	return lo * (hi / lo) ** torch.linspace(0.0, 1.0, n)


def build_2d_spe() -> torch.Tensor:
	"""
	2D sinusoidal PE for IMG_DS×IMG_DS grid.
	Returns [N_IMG_ONE, D_SPE] = [225, 32].
	"""
	periods = _geom_periods(N_FREQS_2D, MIN_P_2D, MAX_P_2D)
	coords  = torch.arange(IMG_DS, dtype=torch.float32)
	ang	 = 2 * math.pi * coords[:, None] / periods[None, :]
	enc1d   = torch.cat([ang.sin(), ang.cos()], dim=1)
	cols = torch.arange(IMG_DS).repeat(IMG_DS)
	rows = torch.arange(IMG_DS).repeat_interleave(IMG_DS)
	return torch.cat([enc1d[cols], enc1d[rows]], dim=1)  # [225, 32]


def build_linear_spe(n_pos: int = N_PROG) -> torch.Tensor:
	"""
	1D sinusoidal PE for n_pos positions.
	Returns [n_pos, D_SPE].
	Default n_pos=N_PROG=96 (covers prog_A and prog_B for Lifter).
	Pass N_PROG_A=48 for ForwardGraphics (one program only).
	"""
	periods = _geom_periods(N_FREQS_LIN, MIN_P_LIN, MAX_P_LIN)
	pos	 = torch.arange(n_pos, dtype=torch.float32)
	ang	 = 2 * math.pi * pos[:, None] / periods[None, :]
	return torch.cat([ang.sin(), ang.cos()], dim=1)  # [n_pos, 32]


# ============================================================================
# Transformer building blocks  (pre-LN, GELU, no proj bias)
# ============================================================================

class SelfAttention(nn.Module):
	"""Full bidirectional multi-head self-attention with FlashAttention dispatch."""
	def __init__(self, d: int, nh: int):
		super().__init__()
		assert d % nh == 0
		self.nh   = nh
		self.dh   = d // nh
		self.qkv  = nn.Linear(d, 3 * d, bias=False)
		self.proj = nn.Linear(d, d,	 bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, T, C = x.shape
		q, k, v = self.qkv(x).split(C, dim=2)
		def rshp(t): return t.view(B, T, self.nh, self.dh).transpose(1, 2)
		q, k, v = rshp(q), rshp(k), rshp(v)
		y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
		return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
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
		x = x + self.mlp(self.ln2(x))
		return x


# ============================================================================
# ForwardGraphics — prog → 32-channel latent image
# ============================================================================

class ForwardGraphics(nn.Module):
	"""
	Forward (generative) model: program tokens → 32-channel latent image.

	Sequence layout: [ img_query×225 | prog×48 ]  (N_FWD_TOK = 273)

	SPE injection (mirror of Lifter — prog processed before image):
		blocks 0..2 : linear PE in dims 0:D_SPE of prog token slots
		blocks 3..5 : 2D PE in dims 0:D_SPE of image token slots

	Input:
		prog_emb: [B, N_PROG_A, d_model]  pre-embedded program tokens.
				  Build with embed_hard(tokens) for discrete tokens, or
				  embed_soft(logits) for a soft distribution from the inverse model.

	Output:
		[B, N_IMG_CH, IMG_H, IMG_W]   (32-channel 30×30 image)
		Channel 0 = grayscale reconstruction; channels 1-31 = latent info.
	"""

	def __init__(
		self,
		n_layers: int = N_LAYERS,
		n_heads:  int = N_HEADS,
		d_model:  int = D_MODEL,
		d_ff:	 int = D_FF,
	):
		super().__init__()
		assert d_model >= D_SPE
		assert n_layers >= 2
		self.n_layers = n_layers
		self.n_heads  = n_heads
		self.d_model  = d_model
		self.n_enc	= n_layers // 2   # blocks that inject prog SPE
		self.d_clean  = d_model - D_SPE

		# Embedding for discrete program tokens
		self.prog_embed = nn.Embedding(VOCAB, d_model)

		# Learned image query tokens (analogous to prog_query in Lifter)
		self.img_query = nn.Parameter(torch.randn(1, N_IMG_ONE, d_model) * 0.02)

		# Transformer
		self.blocks   = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
		self.ln_final = nn.LayerNorm(d_model)

		# Unembedding: image-token SPE-free dims → N_IMG_CH pixel channels
		self.img_unembed  = nn.Linear(self.d_clean, N_IMG_CH, bias=True)

		# Upsample: 15×15 → 30×30
		self.img_upsample = nn.ConvTranspose2d(N_IMG_CH, N_IMG_CH, kernel_size=4, stride=2, padding=1)

		# SPE buffers [1, N_FWD_TOK, d_model]
		#   spe_prog: linear PE in prog slots   (positions N_IMG_ONE:N_FWD_TOK), zeros in image slots
		#   spe_img:  2D PE in image slots	  (positions 0:N_IMG_ONE),		 zeros in prog slots
		spelin = build_linear_spe(N_PROG_A)   # [48, 32]  one program
		spe2d  = build_2d_spe()			   # [225, 32]

		prog_buf = torch.zeros(N_FWD_TOK, d_model)
		prog_buf[N_IMG_ONE:, :D_SPE] = spelin

		img_buf = torch.zeros(N_FWD_TOK, d_model)
		img_buf[:N_IMG_ONE, :D_SPE]  = spe2d

		self.register_buffer('spe_prog', prog_buf.unsqueeze(0))  # [1, 273, d]
		self.register_buffer('spe_img',  img_buf.unsqueeze(0))   # [1, 273, d]

		self.opt = torch.optim.AdamW(self.parameters(), lr=LR_FWD, weight_decay=WD_FWD)

	# -------------------------------------------------------------------------
	def embed_hard(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Embed discrete token indices.  tokens: [B, N_PROG_A] int64."""
		return self.prog_embed(tokens.long())

	def embed_soft(self, logits: torch.Tensor) -> torch.Tensor: 	
		"""
		Soft embedding from inverse-model logit distribution.
		logits: [B, N_PROG_A, VOCAB]  →  [B, N_PROG_A, d_model]
		Uses this model's own prog_embed weight matrix so the embedding space
		is consistent regardless of where the logits came from.
		"""
		return F.softmax(logits, dim=-1) @ self.prog_embed.weight

	# -------------------------------------------------------------------------
	def forward(self, prog_emb: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			prog_emb: [B, N_PROG_A, d_model]

		Returns:
			[B, N_IMG_CH, IMG_H, IMG_W]
		"""
		B = prog_emb.shape[0]
		img_q = self.img_query.expand(B, -1, -1)		  # [B, 225, d]
		h = torch.cat([img_q, prog_emb], dim=1)			# [B, 273, d]

		for l, block in enumerate(self.blocks):
			h = h + (self.spe_prog if l < self.n_enc else self.spe_img)
			h = block(h)

		h = self.ln_final(h)

		# Read image token slots, SPE-free dims (2D SPE was injected there)
		img_h	= h[:, :N_IMG_ONE, D_SPE:]				# [B, 225, d_clean]
		img_flat = self.img_unembed(img_h)				  # [B, 225, N_IMG_CH]
		img_2d   = img_flat.view(B, IMG_DS, IMG_DS, N_IMG_CH).permute(0, 3, 1, 2)  # [B, 32, 15, 15]
		return self.img_upsample(img_2d)					# [B, 32, 30, 30]

	# -------------------------------------------------------------------------
	def train_step(self,
				   prog:	 torch.Tensor,
				   img_real: torch.Tensor) -> Dict[str, float]:
		"""
		Standalone reconstruction training step (no inverse model coupling).

		Args:
			prog:	 [B, N_PROG_A] int64  ground-truth program tokens
			img_real: [B, IMG_H, IMG_W]	float32 ground-truth grayscale image

		Returns dict with 'recon_loss'.
		"""
		self.train()
		prog_emb = self.embed_hard(prog)
		img_hat  = self.forward(prog_emb)					 # [B, 32, H, W]
		loss	 = F.mse_loss(img_hat[:, 0], img_real)		# channel 0 only
		self.opt.zero_grad()
		loss.backward()
		self.opt.step()
		return {'recon_loss': loss.item()}

	# -------------------------------------------------------------------------
	def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
		torch.save({
			'model_state': self.state_dict(),
			'opt_state':   self.opt.state_dict(),
			'step':		step,
		}, path)

	def load_checkpoint(self, path: str) -> Optional[int]:
		ck = torch.load(path, map_location='cpu', weights_only=True)
		self.load_state_dict(ck['model_state'])
		self.opt.load_state_dict(ck['opt_state'])
		return ck.get('step')

	def count_params(self) -> int:
		return sum(p.numel() for p in self.parameters())


# ============================================================================
# Lifter — inverse graphics edit model (img_A + img_B + prog_A → prog_B)
# ============================================================================

class InverseGraphics(nn.Module):
	"""
	Transformer for LOGO inverse graphics *editing*.

	Both images are 32-channel: channel 0 = grayscale, channels 1-31 = latent
	information from ForwardGraphics (or zeros if unavailable).

	Sequence layout: [ img_A×225 | img_B×225 | prog_A×48 | prog_B_query×48 ]

	SPE injection:
		blocks 0..2 : 2D PE in dims 0:D_SPE of image token slots (A and B share
					  the same 15×15 grid)
		blocks 3..5 : linear PE in dims 0:D_SPE of all 96 prog token slots

	Unembedding: prog_B token SPE-free dims → logits [B, 48, VOCAB].
	"""

	def __init__(
		self,
		n_layers: int = N_LAYERS,
		n_heads:  int = N_HEADS,
		d_model:  int = D_MODEL,
		d_ff:	 int = D_FF,
	):
		super().__init__()
		assert d_model >= D_SPE
		assert n_layers >= 2
		self.n_layers = n_layers
		self.n_heads  = n_heads
		self.d_model  = d_model
		self.d_ff	 = d_ff
		self.n_enc	= n_layers // 2
		self.d_clean  = d_model - D_SPE

		# -- Embeddings -------------------------------------------------------
		# 32-channel image → tokens; shared conv+proj for img_A and img_B
		self.img_conv   = nn.Conv2d(N_IMG_CH, 16, kernel_size=4, stride=2, padding=1)
		self.img_proj   = nn.Linear(16, d_model, bias=True)
		self.prog_embed = nn.Embedding(VOCAB, d_model)
		self.prog_query = nn.Parameter(torch.randn(1, N_PROG_B, d_model) * 0.02)

		# -- Transformer ------------------------------------------------------
		self.blocks   = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
		self.ln_final = nn.LayerNorm(d_model)

		# -- Unembedding (SPE-free dims only) ---------------------------------
		self.unembed = nn.Linear(self.d_clean, VOCAB, bias=False)

		# -- SPE buffers [1, N_TOK, d_model] ----------------------------------
		#   spe_enc: 2D PE in image token slots (A and B share the same grid)
		#   spe_dec: linear PE in all 96 prog token slots
		spe2d  = build_2d_spe()		   # [225, 32]
		spelin = build_linear_spe(N_PROG) # [96,  32]

		enc_buf = torch.zeros(N_TOK, d_model)
		enc_buf[:N_IMG_ONE,	  :D_SPE] = spe2d   # img_A
		enc_buf[N_IMG_ONE:N_IMG, :D_SPE] = spe2d   # img_B (same spatial grid)

		dec_buf = torch.zeros(N_TOK, d_model)
		dec_buf[N_IMG:,		  :D_SPE] = spelin  # all 96 prog tokens

		self.register_buffer('spe_enc', enc_buf.unsqueeze(0))  # [1, N_TOK, d]
		self.register_buffer('spe_dec', dec_buf.unsqueeze(0))  # [1, N_TOK, d]

		# -- Optimiser --------------------------------------------------------
		self.weight_opt = torch.optim.AdamW(self.parameters(), lr=LR_W, weight_decay=WD_W)

	# -------------------------------------------------------------------------
	def _embed_img(self, img: torch.Tensor) -> torch.Tensor:
		"""img: [B, N_IMG_CH, H, W]  →  [B, N_IMG_ONE, d_model]"""
		B = img.shape[0]
		f = self.img_conv(img)								  # [B, 16, 15, 15]
		f = f.permute(0, 2, 3, 1).reshape(B, N_IMG_ONE, 16)
		return self.img_proj(f)								 # [B, 225, d]

	# -------------------------------------------------------------------------
	def forward(self,
				img_a:  torch.Tensor,
				img_b:  torch.Tensor,
				prog_a: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			img_a:  [B, N_IMG_CH, H, W]  32-channel image A
			img_b:  [B, N_IMG_CH, H, W]  32-channel image B
			prog_a: [B, N_PROG_A]		 int64 program A tokens

		Returns:
			logits [B, N_PROG_B, VOCAB]
		"""
		B = img_a.shape[0]
		img_a_emb  = self._embed_img(img_a)					  # [B, 225, d]
		img_b_emb  = self._embed_img(img_b)					  # [B, 225, d]
		prog_a_emb = self.prog_embed(prog_a.long())			  # [B,  48, d]
		prog_b_emb = self.prog_query.expand(B, -1, -1)		  # [B,  48, d]

		h = torch.cat([img_a_emb, img_b_emb, prog_a_emb, prog_b_emb], dim=1)  # [B, N_TOK, d]

		for l, block in enumerate(self.blocks):
			h = h + (self.spe_enc if l < self.n_enc else self.spe_dec)
			h = block(h)

		h = self.ln_final(h)
		prog_b_h = h[:, N_IMG + N_PROG_A:, D_SPE:]			  # [B, 48, d_clean]
		return self.unembed(prog_b_h)							# [B, 48, VOCAB]

	# -------------------------------------------------------------------------
	def train_step(self,
				   img_a:  torch.Tensor,
				   img_b:  torch.Tensor,
				   prog_a: torch.Tensor,
				   prog_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""
		Standalone training step (without forward graphics coupling).

		Args:
			img_a, img_b:   [B, N_IMG_CH, H, W]  float32
			prog_a, prog_b: [B, 48]			   int64

		Returns (logits, losses_dict).
		"""
		self.train()
		logits = self.forward(img_a, img_b, prog_a)
		loss   = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))
		self.weight_opt.zero_grad()
		loss.backward()
		self.weight_opt.step()
		return logits, {'loss': loss.item()}

	# -------------------------------------------------------------------------
	def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
		torch.save({
			'model_state':	  self.state_dict(),
			'weight_opt_state': self.weight_opt.state_dict(),
			'step':			 step,
		}, path)

	def load_checkpoint(self, path: str) -> Optional[int]:
		ck = torch.load(path, map_location='cpu', weights_only=True)
		self.load_state_dict(ck['model_state'])
		self.weight_opt.load_state_dict(ck['weight_opt_state'])
		return ck.get('step')

	# -------------------------------------------------------------------------
	def count_params(self) -> Dict[str, int]:
		return {'weight_params': sum(p.numel() for p in self.parameters())}

	def flops_per_forward(self) -> int:
		"""Theoretical FLOPs for one forward pass on one sample (matmuls only)."""
		d, ff, nh = self.d_model, self.d_ff, self.n_heads
		dh, T = d // nh, N_TOK
		# Conv2d(N_IMG_CH,16,k=4,s=2): 2 images
		f  = 2 * 2 * N_IMG_CH * 16 * 4 * 4 * IMG_DS * IMG_DS
		# img_proj: Linear(16→d) over N_IMG tokens
		f += 2 * N_IMG * 16 * d
		block_f = (
			2 * T * d * 3 * d +	 # QKV projection
			4 * nh * T * T * dh +   # attn scores + value aggregation
			2 * T * d * d +		 # output projection
			2 * T * d * ff +		# FFN fc1
			2 * T * ff * d		  # FFN fc2
		)
		f += self.n_layers * block_f
		f += 2 * N_PROG_B * self.d_clean * VOCAB  # unembedding
		return f


# ============================================================================
# CombinedModel — joint training and iterative inference
# ============================================================================

class CombinedModel(nn.Module):
	"""
	Combines ForwardGraphics and InverseGraphics for joint training and iterative inference.

	Training step (x=[B,2,H,W] grayscale, y=[B,96] program tokens):
		1. img_a_fwd = fwd(embed_hard(prog_a))			  [B, 32, H, W]  (once)
		   img_a_32  = cat([x[:,0:1], img_a_fwd[:,1:]])	 ch 0 = pixel, 1-31 = latent
		2. img_b_32  = cat([x[:,1:2], zeros(B,31,H,W)])	 no latent info yet
		3. for k in range(k_inf):
			   logits	= inv(img_a_32, img_b_32, prog_a)
			   img_b_fwd = fwd(embed_soft(logits))
			   img_b_32  = cat([x[:,1:2], img_b_fwd[:,1:]])
		4. loss = CE(logits, prog_b)
				+ lambda_recon * [ MSE(img_a_fwd[:,0], x[:,0])
								 + MSE(fwd(embed_hard(prog_b))[:,0], x[:,1]) ]
		   loss.backward()  ->  inv.weight_opt.step(), fwd.opt.step()

	The gradient graph is fully unrolled across all K iterations.  Both models
	receive gradients from the CE loss (inv directly; fwd via embed_soft) and
	from the reconstruction loss (fwd directly via channel-0 pixel MSE).

	Inference (predict): same iterative loop, no gradients, k_inf overridable.
	"""

	def __init__(
		self,
		inv:		  InverseGraphics,
		fwd:		  ForwardGraphics,
		k_inf:		int   = K_INF,
		lambda_recon: float = LAMBDA_RECON,
	):
		super().__init__()
		self.inv		  = inv
		self.fwd		  = fwd
		self.k_inf		= k_inf
		self.lambda_recon = lambda_recon

	# -------------------------------------------------------------------------
	@staticmethod
	def _compose_img32(gray: torch.Tensor, fwd_out: torch.Tensor) -> torch.Tensor:
		"""
		Build a 32-channel image: ch 0 = grayscale, ch 1-31 = fwd latent.
			gray:	[B, H, W]
			fwd_out: [B, N_IMG_CH, H, W]
		Returns:	[B, N_IMG_CH, H, W]
		"""
		return torch.cat([gray[:, None], fwd_out[:, 1:]], dim=1)

	# -------------------------------------------------------------------------
	def train_step(self,
				   x: torch.Tensor,
				   y: torch.Tensor,
				   forward_only: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""
		Joint training step.

		Args:
			x: [B, 2, H, W]  float32  grayscale images (ch 0 = img_A, ch 1 = img_B)
			y: [B, 96]		int64	y[:,:48] = prog_A, y[:,48:] = prog_B

		Returns (logits [B, 48, VOCAB], img_b_recon [B, H, W], losses_dict).
		img_b_recon is channel 0 (grayscale) of fwd(embed_hard(prog_b)), detached.
		"""
		self.inv.train()
		self.fwd.train()
		B, dev = x.shape[0], x.device

		prog_a = y[:, :N_PROG_A].long()
		prog_b = y[:, N_PROG_A:].long()

		# (1) Forward model on prog_a — computed once, gradients active
		img_a_fwd = self.fwd(self.fwd.embed_hard(prog_a))		 # [B, 32, H, W]
		img_a_32  = self._compose_img32(x[:, 0], img_a_fwd)

		# (2) Initialise img_b_32: grayscale only, no latent yet
		img_b_32 = self._compose_img32(
			x[:, 1],
			torch.zeros(B, N_IMG_CH, IMG_H, IMG_W, device=dev),
		)

		# (3) Iterative refinement — fully unrolled, gradients through all K passes
		if not forward_only:
			for _ in range(self.k_inf):
				logits	= self.inv(img_a_32, img_b_32, prog_a)	   # [B, 48, VOCAB]
				img_b_fwd = self.fwd(self.fwd.embed_soft(logits))	  # [B, 32, H, W]
				img_b_32  = self._compose_img32(x[:, 1], img_b_fwd)

			# (4) Joint loss
			ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))
		else:
			ce_loss = torch.zeros(1)
			logits = torch.zeros(B, 48, VOCAB)

		# Reconstruction: fwd should recover ch-0 pixels from ground-truth progs.
		# img_a_fwd already in graph; fresh pass needed for prog_b (loop used soft).
		img_b_recon = self.fwd(self.fwd.embed_hard(prog_b))		# [B, 32, H, W]

		recon_loss  = (F.mse_loss(img_a_fwd[:, 0, :, :], x[:, 0, :, :]) +
					   F.mse_loss(img_b_recon[:, 0, :, :], x[:, 1, :, :]))

		self.inv.weight_opt.zero_grad()
		self.fwd.opt.zero_grad()

		if not forward_only:
			total_loss = ce_loss + self.lambda_recon * recon_loss
			total_loss.backward()
			self.fwd.opt.step()
			self.inv.weight_opt.step()

			return logits, img_b_recon[:, 0].detach(), {
				'ce_loss':    ce_loss.item(),
				'recon_loss': recon_loss.item(),
				'total_loss': total_loss.item(),
			}
		else: # forward
			recon_loss.backward()
			self.fwd.opt.step()

			return logits, img_b_recon[:, 0].detach(), {
				'ce_loss':    0.0,
				'recon_loss': recon_loss.item(),
				'total_loss': recon_loss.item(),
			}

	# -------------------------------------------------------------------------
	@torch.no_grad()
	def predict(self,
				x: torch.Tensor,
				y: torch.Tensor,
				k_inf:  Optional[int] = None) -> torch.Tensor:
		"""
		Iterative inference — no ground-truth prog_b required.

		Args:
			x:	  [B, 2, H, W]   float32 grayscale images
			y: [B, 96]		int64	y[:,:48] = prog_A, y[:,48:] = prog_B
			k_inf:  iteration count (default: self.k_inf)

		Returns logits [B, N_PROG_B, VOCAB].
		"""
		self.inv.eval()
		self.fwd.eval()
		prog_a = y[:, :N_PROG_A].long()
		prog_b = y[:, N_PROG_A:].long()
		B, dev = x.shape[0], x.device
		K = k_inf if k_inf is not None else self.k_inf

		img_a_fwd = self.fwd(self.fwd.embed_hard(prog_a))
		img_a_32  = self._compose_img32(x[:, 0], img_a_fwd)

		img_b_32 = self._compose_img32(
			x[:, 1],
			torch.zeros(B, N_IMG_CH, IMG_H, IMG_W, device=dev),
		)

		# Pass 0: inv sees img_b with no latent info yet
		logits = self.inv(img_a_32, img_b_32, prog_a)
		ce_loss_pre = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))

		for _ in range(K):
			img_b_recon = self.fwd(self.fwd.embed_soft(logits))
			img_b_32  = self._compose_img32(x[:, 1], img_b_recon)
			logits	= self.inv(img_a_32, img_b_32, prog_a)

		ce_loss_post = F.cross_entropy(logits.reshape(-1, VOCAB), prog_b.reshape(-1))
		return logits, img_b_recon[:,0].detach(), {'ce_loss_pre': ce_loss_pre.item(), 'ce_loss_post': ce_loss_post.item()}

	# -------------------------------------------------------------------------
	def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
		torch.save({
			'inv_model_state': self.inv.state_dict(),
			'inv_opt_state':   self.inv.weight_opt.state_dict(),
			'fwd_model_state': self.fwd.state_dict(),
			'fwd_opt_state':   self.fwd.opt.state_dict(),
			'step':			step,
		}, path)

	def load_checkpoint(self, path: str) -> Optional[int]:
		ck = torch.load(path, map_location='cpu', weights_only=True)
		self.inv.load_state_dict(ck['inv_model_state'])
		self.inv.weight_opt.load_state_dict(ck['inv_opt_state'])
		self.fwd.load_state_dict(ck['fwd_model_state'])
		self.fwd.opt.load_state_dict(ck['fwd_opt_state'])
		return ck.get('step')


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == '__main__':
	torch.manual_seed(42)
	dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {dev}\n")
	B = 4

	# -- ForwardGraphics ------------------------------------------------------
	print("=== ForwardGraphics ===")
	fwd = ForwardGraphics().to(dev)
	prog = torch.randint(0, VOCAB, (B, N_PROG_A), device=dev)

	# Hard (discrete) embedding
	img_out = fwd(fwd.embed_hard(prog))
	assert img_out.shape == (B, N_IMG_CH, IMG_H, IMG_W), img_out.shape
	print(f"  Output shape (hard):  {tuple(img_out.shape)}  OK")

	# Soft embedding from fake logits
	logits_fake = torch.randn(B, N_PROG_A, VOCAB, device=dev)
	img_soft = fwd(fwd.embed_soft(logits_fake))
	assert img_soft.shape == (B, N_IMG_CH, IMG_H, IMG_W)
	print(f"  Output shape (soft):  {tuple(img_soft.shape)}  OK")
	print(f"  Params: {fwd.count_params():,}")

	img_real = torch.rand(B, IMG_H, IMG_W, device=dev)
	losses_f = fwd.train_step(prog, img_real)
	print(f"  Recon loss:		   {losses_f['recon_loss']:.4f}")

	# -- InverseGraphics ---------------------------------------------------------------
	print("\n=== InverseGraphics ===")
	inv = InverseGraphics().to(dev)
	img_a = torch.rand(B, N_IMG_CH, IMG_H, IMG_W, device=dev)
	img_b = torch.rand(B, N_IMG_CH, IMG_H, IMG_W, device=dev)
	prog_a = torch.randint(0, VOCAB, (B, N_PROG_A), device=dev)
	prog_b = torch.randint(0, VOCAB, (B, N_PROG_B), device=dev)

	logits = inv(img_a, img_b, prog_a)
	assert logits.shape == (B, N_PROG_B, VOCAB), logits.shape
	print(f"  Logits shape:		 {tuple(logits.shape)}  OK")
	print(f"  Params:			   {inv.count_params()['weight_params']:,}")
	print(f"  FLOPs/fwd/sample:	 {inv.flops_per_forward()/1e9:.3f} GFLOPs")

	logits2, losses_i = inv.train_step(img_a, img_b, prog_a, prog_b)
	print(f"  Train loss:		   {losses_i['loss']:.4f}")

	# -- CombinedModel --------------------------------------------------------
	print("\n=== CombinedModel ===")
	x = torch.rand(B, 2, IMG_H, IMG_W, device=dev)   # two grayscale images
	y = torch.randint(0, VOCAB, (B, N_PROG), device=dev)

	combined = CombinedModel(inv, fwd)

	logits_c, img_b_recon_c, losses_c = combined.train_step(x, y)
	assert logits_c.shape == (B, N_PROG_B, VOCAB)
	assert img_b_recon_c.shape == (B, IMG_H, IMG_W)
	print(f"  img_b_recon shape:    {tuple(img_b_recon_c.shape)}  OK")
	print(f"  train_step logits:	{tuple(logits_c.shape)}  OK")
	print(f"  ce_loss:			  {losses_c['ce_loss']:.4f}")
	print(f"  recon_loss:		   {losses_c['recon_loss']:.4f}")
	print(f"  total_loss:		   {losses_c['total_loss']:.4f}")

	prog_a_inf = y[:, :N_PROG_A]
	logits_inf = combined.predict(x, prog_a_inf)
	assert logits_inf.shape == (B, N_PROG_B, VOCAB)
	print(f"  predict logits:	   {tuple(logits_inf.shape)}  OK")

	import tempfile, os
	with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
		ckpt_path = f.name
	combined.save_checkpoint(ckpt_path, step=42)
	step = combined.load_checkpoint(ckpt_path)
	os.unlink(ckpt_path)
	assert step == 42
	print(f"  checkpoint round-trip step={step}  OK")
