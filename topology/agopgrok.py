import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. Network Definition
# ==========================================
class GrokkingMLP(nn.Module):
	def __init__(self, p, d):
		super().__init__()
		self.E_A = nn.Embedding(p, d)
		self.E_B = nn.Embedding(p, d)

		nn.init.normal_(self.E_A.weight, mean=0.0, std=1.0)
		nn.init.normal_(self.E_B.weight, mean=0.0, std=1.0)

		self.W_out = nn.Linear(d, p, bias=False)
		nn.init.normal_(self.W_out.weight, mean=0.0, std=1.0 / np.sqrt(d))

	def forward(self, a, b):
		h = (self.E_A(a) + self.E_B(b)) ** 2
		return self.W_out(h)

# ==========================================
# 2. Exact Analytical AGOP Laplacian
# ==========================================
def compute_analytical_laplacian(E_A, E_B, W_out, batch_a, batch_b):
	"""
	Computes the exact Average Gradient Outer Product (AGOP) of the logits
	with respect to the 2p one-hot inputs analytically, bypassing autograd.
	"""
	# 1. Forward Pass hidden features
	v = E_A[batch_a] + E_B[batch_b] # (Batch, d)
	# indexing is faster than multply...

	# 2. Derivative of h = v^2 is g = 2v
	g = 2 * v # (Batch, d)

	# 3. Compute V = Expectation of (g * g^T) over the batch
	# torch.einsum is a fast, vectorized way to do batch-wise outer products
	V = torch.mean(torch.einsum('bi,bj->bij', g, g), dim=0) # (d, d)
	g_mean = torch.mean(g, dim=0)
	V_cov = V - torch.outer(g_mean, g_mean)

	# 4. The exact Jacobian Outer Product Matrix for the hidden layer
	# M = (W_out^T @ W_out) element-wise multiplied by V
	W_T_W = W_out.t() @ W_out # (d, d)
	M = W_T_W * V_cov

	# 5. Project back into the 2p x 2p one-hot input space
	E_joint = torch.cat([E_A, E_B], dim=0) # (2p, d)
	G = E_joint @ M @ E_joint.t() # (2p, 2p) This is the EXACT AGOP Matrix!

	# 6. Normalize and apply Geometric Clamp (Fixing the folded manifold!)
	# We only want positive correlations to act as attractive springs.
	G_norm = G / (torch.max(torch.abs(G)) + 1e-8)
	W = torch.clamp(G_norm, min=0.0)

	# 7. Compute Laplacian (Detach it to prevent trivial collapse shortcuts!)
	D = torch.diag(W.sum(dim=1))
	L = (D - W).detach()

	return L, E_joint

def compute_analytical_agop(E_A, E_B, W_out, batch_a, batch_b):
	# 1. Forward Pass
	v = E_A[batch_a] + E_B[batch_b]
	g = 2 * v

	# 2. Compute true Covariance of the gradients
	# Var(X) = E[X*X^T] - E[X]*E[X]^T
	V_uncentered = torch.mean(torch.einsum('bi,bj->bij', g, g), dim=0)
	g_mean = torch.mean(g, dim=0)
	V_cov = V_uncentered - torch.outer(g_mean, g_mean) # <--- THE FIX

	# 3. Exact Centered Jacobian Outer Product
	W_T_W = W_out.t() @ W_out
	M_centered = W_T_W * V_cov

	# 4. Project to Token Space
	E_joint = torch.cat([E_A, E_B], dim=0)
	G_cov = E_joint @ M_centered @ E_joint.t()

	# 5. The Paper's Exact Regularizer: Trace of the AGOP
	# We do not use D - W. We just return the trace!
	agop_trace_penalty = torch.trace(G_cov)

	return agop_trace_penalty

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(p=59, d=128, epochs=1000, use_agop_loss=False, device='cpu'):
	dataset, labels = [],[]
	for a in range(p):
		for b in range(p):
			dataset.append([a, b])
			labels.append((a + b) % p)

	dataset = torch.tensor(dataset, dtype=torch.long)
	labels = torch.tensor(labels, dtype=torch.long)

	torch.manual_seed(42)
	indices = torch.randperm(p * p)
	split_idx = int(0.75 * p * p)
	train_idx, val_idx = indices[:split_idx], indices[split_idx:]

	train_data, val_data = dataset[train_idx].to(device), dataset[val_idx].to(device)
	train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

	model = GrokkingMLP(p, d).to(device)

	# Standard Weight Decay acts as the isotropic contraction
	optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
	criterion = nn.CrossEntropyLoss()

	history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()

		logits = model(train_data[:, 0], train_data[:, 1])
		loss = criterion(logits, train_labels)

		# --- The Ultimate Force: Analytical AGOP Topology ---
		if use_agop_loss:
			if False:
				L, E_joint = compute_analytical_laplacian(
					model.E_A.weight, model.E_B.weight, model.W_out.weight,
					train_data[:, 0], train_data[:, 1]
				)

				# Apply the geometric Laplacian tension
				topo_loss = torch.trace(E_joint.t() @ L @ E_joint)
			else:
				topo_loss = compute_analytical_agop(
					model.E_A.weight, model.E_B.weight, model.W_out.weight,
					train_data[:, 0], train_data[:, 1]
				)

			lambda_topo = 1.0 / (2 * p * d)
			loss = loss + lambda_topo * topo_loss

		loss.backward()
		optimizer.step()

		if epoch % 10 == 0:
			model.eval()
			with torch.no_grad():
				val_logits = model(val_data[:, 0], val_data[:, 1])
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
	if torch.cuda.is_available(): device = torch.device("cuda")
	elif torch.backends.mps.is_available(): device = torch.device("mps")
	else: device = torch.device("cpu")
	print(f"Using device: {device}")
	epochs = 500

	print("Training Standard MLP...")
	std_model, std_hist = train_model(epochs=epochs, use_agop_loss=False, device=device)

	print("Training Analytical AGOP-Assisted MLP...")
	topo_model, topo_hist = train_model(epochs=epochs, use_agop_loss=True, device=device)

	# --- Plot 1: Training Curves ---
	fig1 = plt.figure(figsize=(8, 5))
	epochs_x = np.arange(0, epochs, 10)
	plt.plot(epochs_x, std_hist['val_acc'], label='Standard MLP', color='red', alpha=0.7)
	plt.plot(epochs_x, topo_hist['val_acc'], label='Analytical AGOP Topo', color='blue', linewidth=2)
	plt.title("Validation Accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.show()

	# --- Plot 2: Visualizing the Circulant Structure ---
	# We reconstruct the AGOP Kernel for both models to prove the topological sorting
	def get_agop_matrix(model):
		model.eval()
		with torch.no_grad():
			# Generate full table
			p = model.E_A.weight.shape[0]
			dataset = torch.tensor([[a, b] for a in range(p) for b in range(p)]).to(device)

			# 1. Compute M (The interaction core)
			v = model.E_A(dataset[:, 0]) + model.E_B(dataset[:, 1])
			g = 2 * v
			V = torch.mean(torch.einsum('bi,bj->bij', g, g), dim=0)
			g_mean = torch.mean(g, dim=0)
			V_cov = V - torch.outer(g_mean, g_mean)

			W_T_W = model.W_out.weight.t() @ model.W_out.weight
			M_centered = W_T_W * V_cov

			# 2. Compute G_A = E_A @ M @ E_A.T (The token-to-token similarity kernel)
			G_A = model.E_A.weight @ M_centered @ model.E_A.weight.t()
			return G_A.cpu().numpy()

	std_G_A = get_agop_matrix(std_model)
	topo_G_A = get_agop_matrix(topo_model)

	fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

	# Plot Standard Model's Kernel
	im1 = axes[0].imshow(std_G_A, cmap='magma')
	axes[0].set_title("Standard MLP: AGOP Matrix $G_A$\n(Unstructured / Noisy)")
	axes[0].set_xlabel("Token Identity")
	axes[0].set_ylabel("Token Identity")
	fig2.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

	# Plot AGOP-Assisted Model's Kernel
	im2 = axes[1].imshow(topo_G_A, cmap='magma')
	axes[1].set_title("AGOP-Assisted MLP: AGOP Matrix $G_A$\n(Perfectly Block-Circulant!)")
	axes[1].set_xlabel("Token Identity")
	axes[1].set_ylabel("Token Identity")
	fig2.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    run_experiment()
