import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# 2. Dynamic AGOP Topological Force
# ==========================================
def compute_dynamic_laplacian_loss(embedding_matrix, ce_loss):
    """
    Computes the AGOP-based Topological Loss dynamically from the current gradients.
    """
    # 1. Extract the gradient of the loss w.r.t the embedding matrix.
    # create_graph=True is the magic that allows us to backprop THROUGH this gradient penalty.
    grad_E = torch.autograd.grad(ce_loss, embedding_matrix, 
                                 create_graph=True, retain_graph=True)[0]
    
    # 2. Center the gradients (Zero-mean along the token dimension)
    grad_centered = grad_E - grad_E.mean(dim=0, keepdim=True)
    
    # 3. L2 Normalize to get pure direction (Cosine Similarity)
    # This prevents the topology loss from exploding if the CE gradients get large
    grad_norm = F.normalize(grad_centered, p=2, dim=1)
    
    # 4. Compute the Covariance / AGOP Matrix
    C = grad_norm @ grad_norm.t()
    
    # 5. Element-wise Square to guarantee Non-Negative Affinity (W >= 0)
    W = C ** 2 
    
    # 6. Compute Graph Laplacian
    D = torch.diag(W.sum(dim=1))
    L = D - W
    
    # 7. Compute Trace Penalty: Tr(E^T L E)
    topo_loss = torch.trace(embedding_matrix.t() @ L @ embedding_matrix)
    
    return topo_loss

# ==========================================
# 3. Training Loop
# ==========================================
def train_model(p=59, d=128, epochs=1000, use_dynamic_agop=False, device='cpu'):
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
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_data[:, 0], train_data[:, 1])
        ce_loss = criterion(logits, train_labels)
        
        loss = ce_loss
        
        # --- The Ultimate Algorithm: Dynamic AGOP Laplacian ---
        if use_dynamic_agop:
            # We compute the geometric pulling force dynamically using the network's own tangents
            topo_loss_A = compute_dynamic_laplacian_loss(model.E_A.weight, ce_loss)
            topo_loss_B = compute_dynamic_laplacian_loss(model.E_B.weight, ce_loss)
            
            # Lambda scales the strength of the topological springs
            lambda_topo = 0.05 / (p * d)
            loss = loss + lambda_topo * (topo_loss_A + topo_loss_B)
            
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
    
    print("Training Standard MLP...")
    std_model, std_hist = train_model(epochs=10000, use_dynamic_agop=False, device=device)
    
    print("Training Dynamic AGOP-Laplacian MLP...")
    topo_model, topo_hist = train_model(epochs=10000, use_dynamic_agop=True, device=device)
    
    fig = plt.figure(figsize=(16, 6))
    epochs_x = np.arange(0, 10000, 10)
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(epochs_x, std_hist['val_acc'], label='Standard MLP', color='red', alpha=0.7)
    ax1.plot(epochs_x, topo_hist['val_acc'], label='Dynamic AGOP Topo', color='blue', linewidth=2)
    ax1.set_title("Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    pca = PCA(n_components=2)
    
    def plot_pca(model, ax, title):
        emb_B = model.E_B.weight.detach().cpu().numpy()
        pca_B = pca.fit_transform(emb_B)
        sc = ax.scatter(pca_B[:, 0], pca_B[:, 1], c=np.arange(59), cmap='twilight_shifted', s=60, edgecolors='k')
        ax.set_title(title)
        return sc
        
    ax2 = plt.subplot(1, 3, 2)
    plot_pca(std_model, ax2, "Standard MLP (E_B)")
    
    ax3 = plt.subplot(1, 3, 3)
    sc3 = plot_pca(topo_model, ax3, "Dynamic AGOP-Laplacian MLP (E_B)\n(Self-Organized from Gradients)")
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc3, cax=cbar_ax, label="Token Identity (0-58)")
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

if __name__ == "__main__":
    run_experiment()
