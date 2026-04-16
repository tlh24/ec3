import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. Network Definition & Activation
# ==========================================
class GrokkingMLP(nn.Module):
	def __init__(self, p, d):
		super().__init__()
		# Embeddings for a and b
		self.E_A = nn.Embedding(p, d)
		self.E_B = nn.Embedding(p, d)

		# Initialize with standard normal (null prior)
		nn.init.normal_(self.E_A.weight, mean=0.0, std=1.0)
		nn.init.normal_(self.E_B.weight, mean=0.0, std=1.0)

		# Linear projection to logits
		self.W_out = nn.Linear(d, p, bias=False)
		nn.init.normal_(self.W_out.weight, mean=0.0, std=1.0 / np.sqrt(d))

	def forward(self, a, b):
		ea = self.E_A(a)
		eb = self.E_B(b)

		# The x^2 Activation Function
		h = (ea + eb) ** 2

		logits = self.W_out(h)
		return logits

# ==========================================
# 2. Bipartite Graph Laplacian (Coupled A and B)
# ==========================================
def get_bipartite_laplacian(p, a0=0, a1=1, b0=0, b1=1, a_cross=0, b_cross=0):
	"""
	Constructs a 2p x 2p Adjacency Matrix locking A and B together.
	Indices 0 to p-1 belong to A, indices p to 2p-1 belong to B.
	"""
	# 1. W_B: Internal ring for B
	P_a0, P_a1 = np.zeros((p, p)), np.zeros((p, p))
	for b in range(p):
		P_a0[b, (a0 + b) % p] = 1
		P_a1[b, (a1 + b) % p] = 1
	W_B = P_a0 @ P_a1.T
	W_B = np.clip(W_B + W_B.T, 0, 1)

	# 2. W_A: Internal ring for A
	P_b0, P_b1 = np.zeros((p, p)), np.zeros((p, p))
	for a in range(p):
		P_b0[a, (a + b0) % p] = 1
		P_b1[a, (a + b1) % p] = 1
	W_A = P_b0 @ P_b1.T
	W_A = np.clip(W_A + W_A.T, 0, 1)

	# 3. W_AB: Cross-connections locking A to B
	# Connects a and b if they produce the same output under neutral anchors
	P_Across, P_Bcross = np.zeros((p, p)), np.zeros((p, p))
	for x in range(p):
		P_Across[x, (x + b_cross) % p] = 1
		P_Bcross[x, (a_cross + x) % p] = 1
	W_AB = P_Across @ P_Bcross.T
	# No need to make symmetric here, we will transpose it into the block matrix

	# 4. Assemble the Joint 2p x 2p Adjacency Matrix
	W_joint = np.zeros((2 * p, 2 * p))
	W_joint[0:p, 0:p] = W_A               # Top-Left: A internal
	W_joint[p:2*p, p:2*p] = W_B           # Bottom-Right: B internal
	W_joint[0:p, p:2*p] = W_AB            # Top-Right: A to B coupling
	W_joint[p:2*p, 0:p] = W_AB.T          # Bottom-Left: B to A coupling

	# 5. Compute Joint Laplacian (L = D - W)
	D = np.diag(W_joint.sum(axis=1))
	L = D - W_joint

	return torch.tensor(L, dtype=torch.float32)

def train_model(p=59, d=128, epochs=1000, use_topo_loss=False, device='cpu'):
	dataset, labels = [],[]
	for a in range(p):
		for b in range(p):
			dataset.append([a, b])
			labels.append((a + b) % p)

	dataset = torch.tensor(dataset, dtype=torch.long)
	labels = torch.tensor(labels, dtype=torch.long)

	# torch.manual_seed(42)
	indices = torch.randperm(p * p)
	split_idx = int(0.75 * p * p)
	train_idx, val_idx = indices[:split_idx], indices[split_idx:]

	train_data, val_data = dataset[train_idx].to(device), dataset[val_idx].to(device)
	train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

	# Fetch the 2p x 2p Joint Laplacian
	L_joint = get_bipartite_laplacian(p).to(device)

	model = GrokkingMLP(p, d).to(device)

	optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
	criterion = nn.CrossEntropyLoss()

	history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()

		logits = model(train_data[:, 0], train_data[:, 1])
		loss = criterion(logits, train_labels)

		# --- Bipartite Topological Forcing ---
		if use_topo_loss:
			# Concatenate E_A and E_B into a single 2p x d tensor
			E_joint = torch.cat([model.E_A.weight, model.E_B.weight], dim=0)

			# Apply the joint Laplacian force in one single trace!
			topo_loss = torch.trace(E_joint.t() @ L_joint @ E_joint)

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

				history['train_loss'].append(loss.item()) # recording total loss
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
	epochs = 2500
	print(f"Using device: {device}")

	print("Training Standard MLP...")
	std_model, std_hist = train_model(epochs=epochs, use_topo_loss=False, device=device)

	print("Training Bipartite Topo-Assisted MLP...")
	topo_model, topo_hist = train_model(epochs=epochs, use_topo_loss=True, device=device)

	# --- Plotting ---
	fig = plt.figure(figsize=(16, 6))
	epochs_x = np.arange(0, epochs, 10)

	# 1. Validation Accuracy
	ax1 = plt.subplot(1, 3, 1)
	ax1.plot(epochs_x, std_hist['val_acc'], label='Standard MLP', color='red', alpha=0.7)
	ax1.plot(epochs_x, topo_hist['val_acc'], label='Bipartite Topo MLP', color='blue', linewidth=2)
	ax1.set_title("Validation Accuracy")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Accuracy")
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# --- PCA Joint Plotting ---
	# We fit the PCA on the combined space of A and B to ensure axes align
	pca = PCA(n_components=2)

	def plot_pca_joint(model, ax, title):
		emb_A = model.E_A.weight.detach().cpu().numpy()
		emb_B = model.E_B.weight.detach().cpu().numpy()

		# Fit PCA on the joint embeddings to capture the shared space
		joint_emb = np.concatenate([emb_A, emb_B], axis=0)
		pca.fit(joint_emb)

		pca_A = pca.transform(emb_A)
		pca_B = pca.transform(emb_B)

		# Plot A (Circles) and B (Crosses)
		sc_A = ax.scatter(pca_A[:, 0], pca_A[:, 1], c=np.arange(59), cmap='twilight_shifted', marker='o', s=60, edgecolors='k', label="E_A (Circles)")
		sc_B = ax.scatter(pca_B[:, 0], pca_B[:, 1], c=np.arange(59), cmap='twilight_shifted', marker='X', s=60, edgecolors='k', label="E_B (Crosses)")

		ax.set_title(title)
		ax.legend(loc='upper right', fontsize=8)
		return sc_A

	ax2 = plt.subplot(1, 3, 2)
	plot_pca_joint(std_model, ax2, "Standard MLP Embeddings")

	ax3 = plt.subplot(1, 3, 3)
	sc3 = plot_pca_joint(topo_model, ax3, "Bipartite Topo MLP Embeddings)")

	# Add a single colorbar for reference
	cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
	fig.colorbar(sc3, cax=cbar_ax, label="Token Identity (0-58)")

	plt.tight_layout(rect=[0, 0, 0.9, 1])
	plt.show()

if __name__ == "__main__":
    run_experiment()
