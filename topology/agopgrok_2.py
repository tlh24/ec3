import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, jacrev
import pdb

USE_NORM = False
BATCH_SIZE = 128

# ==========================================
# 1. Network Definition
# ==========================================
class GrokkingTransformer(nn.Module):
	def __init__(self, p, d, n_heads=1):
		super().__init__()
		assert d % n_heads == 0, "d must be divisible by n_heads"
		self.p = p
		self.d = d
		self.n_heads = n_heads
		self.head_dim = d // n_heads

		# Vocab: 0 to p-1 are standard numbers. Token `p` is the special '=' token.
		self.embed = nn.Embedding(p + 1, d)
		nn.init.normal_(self.embed.weight, std=0.02)

		self.pos_emb = nn.Embedding(3, d)
		nn.init.normal_(self.pos_emb.weight, std=0.02)

		# Multi-head attention projections
		self.W_q = nn.Linear(d, d, bias=False)
		self.W_k = nn.Linear(d, d, bias=False)
		self.W_v = nn.Linear(d, d, bias=False)
		self.W_o = nn.Linear(d, d, bias=False)

		self.ln1 = nn.LayerNorm(d) if USE_NORM else nn.Identity()
		self.ln2 = nn.LayerNorm(d) if USE_NORM else nn.Identity()

		# Quadratic MLP (highly favored for grokking modular arithmetic phases)
		self.mlp_w1 = nn.Linear(d, 4 * d, bias=True)
		self.mlp_w2 = nn.Linear(4 * d, d, bias=True)

		self.W_out = nn.Linear(d, p, bias=True)
		nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(d))

	def forward(self, x):
		# x shape: (B, 3)
		e = self.embed(x)
		return self.forward_from_embeddings(e)

	def forward_from_embeddings(self, e):
		# Unroll logic allows torch.func.vmap to run over the sequence
		# pdb.set_trace()
		is_batched = e.dim() == 3
		if not is_batched: e = e.unsqueeze(0)

		B, seq_len, d = e.shape

		# Inject Position Information
		positions = torch.arange(seq_len, device=e.device)
		e = e + self.pos_emb(positions)

		x_norm = self.ln1(e)
		q, k, v = self.W_q(x_norm), self.W_k(x_norm), self.W_v(x_norm)

		# Split into heads: (B, seq, d) -> (B*n_heads, seq, head_dim)
		def split_heads(t):
			return t.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(B * self.n_heads, seq_len, self.head_dim)

		q, k, v = split_heads(q), split_heads(k), split_heads(v)

		scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dim)

		# Standard causal mask for autoregressive emulation
		# not strictly required..
		mask = torch.tril(torch.ones(seq_len, seq_len, device=e.device)).unsqueeze(0)
		scores = scores.masked_fill(mask == 0, float('-inf'))
		attn = torch.softmax(scores, dim=-1)
		# # Manually create the perfect attention map for the '=' token (pos 2)
		# # It attends equally to pos 0 (a) and pos 1 (b)
		# attn = torch.zeros(B, 3, 3, device=e.device)
		# attn[:, 2, 0] = 0.5
		# attn[:, 2, 1] = 0.5
		# attn[:, 2, 2] = 0.0

		# Merge heads back: (B*n_heads, seq, head_dim) -> (B, seq, d)
		context = torch.bmm(attn, v).view(B, self.n_heads, seq_len, self.head_dim).transpose(1, 2).reshape(B, seq_len, d)
		# context = self.W_o(context) # redundant
		h = e + context

		# Pre-LN Quadratic MLP
		h_norm = self.ln2(context)
		mlp_out = self.mlp_w2(nn.functional.relu(self.mlp_w1(h_norm)) )
		h = h + mlp_out

		# We only predict from the last token '=' (position index 2)
		logits = self.W_out(h[:, -1, :])

		if not is_batched: logits = logits.squeeze(0)
		return logits

# ==========================================
# 2. Functional AGOP Implementation
# ==========================================
def compute_agop_transformer(model, batch_data):
	# Retrieve base embeddings
	e = model.embed(batch_data) # Shape: (B, 3, d)

	# Isolate the function we want the Jacobian of
	def single_forward(e_single):
		return model.forward_from_embeddings(e_single)

	# MAGIC HAPPENS HERE: Batched exact Jacobian extraction
	# Shape of J: (B, p, 3, d) -> [Batch, Logits, Seq_len, Emb_Dim]
	J = vmap(jacrev(single_forward))(e)

	# Extract Jacobian traces for token A (pos 0) and token B (pos 1)
	J_a, J_b = J[:, :, 0, :], J[:, :, 1, :]

	# Center the Jacobians to extract true Covariance
	J_a_centered = J_a - J_a.mean(dim=0, keepdim=True)
	J_b_centered = J_b - J_b.mean(dim=0, keepdim=True)

	B = e.shape[0]
	# Expected Outer Product equivalent to original (W^T W) * V_cov
	M_a = torch.einsum('bpi,bpj->ij', J_a_centered, J_a_centered) / B
	M_b = torch.einsum('bpi,bpj->ij', J_b_centered, J_b_centered) / B

	# Project back to pure token space (excluding the '=' token)
	E_num = model.embed.weight[:-1] # Shape: (p, d)

	G_a = E_num @ M_a @ E_num.t()
	G_b = E_num @ M_b @ E_num.t()

	return torch.trace(G_a) + torch.trace(G_b)

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(p=59, d=128, epochs=1000, use_agop_loss=False, device='cpu', batch_size=None):
	dataset, labels = [], []
	for a in range(p):
		for b in range(p):
			dataset.append([a, b, p]) # p is the '=' token index
			labels.append((a + b) % p)

	dataset, labels = torch.tensor(dataset, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

	indices = torch.randperm(p * p)
	split_idx = int(0.6 * p * p)
	train_idx, val_idx = indices[:split_idx], indices[split_idx:]

	train_data, val_data = dataset[train_idx].to(device), dataset[val_idx].to(device)
	train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

	n_train = train_data.shape[0]
	use_minibatch = batch_size is not None and batch_size < n_train

	# steps_per_epoch: gradient steps per effective epoch (full-dataset pass equivalent)
	steps_per_epoch = n_train // batch_size if use_minibatch else 1
	total_steps = epochs * steps_per_epoch
	log_every = 10 * steps_per_epoch  # log every 10 effective epochs

	model = GrokkingTransformer(p, d).to(device)

	# 1e-3 LR is usually safer than 1e-2 for Transformers
	optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
	criterion = nn.CrossEntropyLoss()
	history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

	for step in range(total_steps):
		model.train()
		optimizer.zero_grad()

		if use_minibatch:
			mb_idx = torch.randperm(n_train, device=device)[:batch_size]
			batch_data = train_data[mb_idx]
			batch_labels = train_labels[mb_idx]
		else:
			batch_data = train_data
			batch_labels = train_labels

		logits = model(batch_data)
		loss = criterion(logits, batch_labels)

		if use_agop_loss:
			topo_loss = compute_agop_transformer(model, batch_data)
			loss = loss + (0.5 / (2 * p * d)) * topo_loss

		loss.backward()
		optimizer.step()

		if step % log_every == 0:
			model.eval()
			with torch.no_grad():
				val_logits = model(val_data)
				v_loss = criterion(val_logits, val_labels).item()
				preds = torch.argmax(val_logits, dim=1)
				v_acc = (preds == val_labels).float().mean().item()

				history['train_loss'].append(loss.item())
				history['val_loss'].append(v_loss)
				history['val_acc'].append(v_acc)

	return model, history

# ==========================================
# 4. Execution & Visualization
# ==========================================
def run_experiment():
	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
	print(f"Using device: {device}")
	epochs = 500

	print("Training Standard Transformer...")
	std_model, std_hist = train_model(epochs=epochs, use_agop_loss=False, device=device, batch_size=BATCH_SIZE)

	print("Training Analytical AGOP-Assisted Transformer...")
	topo_model, topo_hist = train_model(epochs=epochs, use_agop_loss=True, device=device, batch_size=BATCH_SIZE)

	# Plot 1: Curves
	epochs_x = np.arange(0, epochs, 10)
	epochs_x4 = np.arange(0, epochs*4, 10) # this is a second variable b/c the vanilla transformer takes longer to train

	fig1, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

	ax_acc.plot(epochs_x4, std_hist['val_acc'], label='Standard Transformer', color='red')
	ax_acc.plot(epochs_x, topo_hist['val_acc'], label='Analytical AGOP Topo', color='blue', linewidth=2)
	ax_acc.set_title("Validation Accuracy")
	ax_acc.set_xlabel("Effective Epochs")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend()
	ax_acc.grid(True, alpha=0.3)

	ax_loss.plot(epochs_x4, std_hist['train_loss'], label='Standard Train', color='salmon', linestyle='--')
	ax_loss.plot(epochs_x4, std_hist['val_loss'], label='Standard Val', color='red')
	ax_loss.plot(epochs_x, topo_hist['train_loss'], label='AGOP Train', color='cornflowerblue', linestyle='--')
	ax_loss.plot(epochs_x, topo_hist['val_loss'], label='AGOP Val', color='blue', linewidth=2)
	ax_loss.set_title("Loss")
	ax_loss.set_xlabel("Effective Epochs")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend()
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()

	# Plot 2: Visualizing Structure
	def get_agop_matrix(model):
		model.eval()
		p = model.p
		dataset = torch.tensor([[a, b, p] for a in range(p) for b in range(p)]).to(device)

		# DO NOT use torch.no_grad() here, jacrev relies on the autograd graph implicitly!
		e = model.embed(dataset)
		J = vmap(jacrev(lambda x: model.forward_from_embeddings(x)))(e)

		J_a = J[:, :, 0, :]
		J_a_centered = J_a - J_a.mean(dim=0, keepdim=True)
		M_a = torch.einsum('bpi,bpj->ij', J_a_centered, J_a_centered) / dataset.shape[0]

		E_num = model.embed.weight[:-1]
		G_A = (E_num @ M_a @ E_num.t()).cpu().detach().numpy()
		return G_A

	fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

	im1 = axes[0].imshow(get_agop_matrix(std_model), cmap='magma')
	axes[0].set_title("Standard Transformer Kernel $G_A$\n(Unstructured / Noisy)")
	fig2.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

	im2 = axes[1].imshow(get_agop_matrix(topo_model), cmap='magma')
	axes[1].set_title("AGOP Transformer Kernel $G_A$\n(Perfectly Block-Circulant!)")
	fig2.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    run_experiment()
