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
# 2. Mathematical Graph Laplacian
# ==========================================
def get_supervised_laplacian(p, anchor0, anchor1):
    """
    Builds the explicit adjacency matrix using the true Cayley Table rules.
    """
    P0 = np.zeros((p, p))
    P1 = np.zeros((p, p))
    for x in range(p):
        P0[x, (anchor0 + x) % p] = 1
        P1[x, (anchor1 + x) % p] = 1
        
    W = P0 @ P1.T
    W = W + W.T  # Make it symmetric (undirected)
    
    D = np.diag(W.sum(axis=1))
    L = D - W
    return torch.tensor(L, dtype=torch.float32)

# ==========================================
# 3. Training Loop (GPU Enabled)
# ==========================================
def train_model(p=59, d=128, epochs=1000, use_topo_loss=False, device='cpu'):
    # --- Data Generation ---
    dataset, labels = [],[]
    for a in range(p):
        for b in range(p):
            dataset.append([a, b])
            labels.append((a + b) % p)
            
    dataset = torch.tensor(dataset, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # Random split
    torch.manual_seed(42)
    indices = torch.randperm(p * p)
    split_idx = int(0.75 * p * p)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    # Push data to GPU
    train_data, val_data = dataset[train_idx].to(device), dataset[val_idx].to(device)
    train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

    # --- Setup Model & Static Laplacians ---
    L_A = get_supervised_laplacian(p, 0, 1).to(device) 
    L_B = get_supervised_laplacian(p, 0, 1).to(device) 
    
    # Push model to GPU
    model = GrokkingMLP(p, d).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Standard Forward Pass
        logits = model(train_data[:, 0], train_data[:, 1])
        ce_loss = criterion(logits, train_labels)
        
        loss = ce_loss
        
        # --- The Frontier: Continuous Topological Forcing ---
        if use_topo_loss:
            # Tr(X^T L X) applied on the GPU
            topo_A = torch.trace(model.E_A.weight.t() @ L_A @ model.E_A.weight)
            topo_B = torch.trace(model.E_B.weight.t() @ L_B @ model.E_B.weight)
            
            lambda_topo = 1.0 / (p * d) 
            loss = loss + lambda_topo * (topo_A + topo_B)
            
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data[:, 0], val_data[:, 1])
                v_loss = criterion(val_logits, val_labels).item()
                preds = torch.argmax(val_logits, dim=1)
                v_acc = (preds == val_labels).float().mean().item()
                
                history['train_loss'].append(ce_loss.item())
                history['val_loss'].append(v_loss)
                history['val_acc'].append(v_acc)
                
    return model, history

# ==========================================
# 4. Execution & Visualization
# ==========================================
def run_experiment():
    # Detect GPU (CUDA for NVIDIA, MPS for Apple Silicon, fallback to CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    epochs = 1500
    
    print("Training Standard MLP (Waiting for Grokking...)")
    std_model, std_hist = train_model(epochs=epochs, use_topo_loss=False, device=device)
    
    print("Training MLP with Topological Force...")
    topo_model, topo_hist = train_model(epochs=epochs, use_topo_loss=True, device=device)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 10))
    epochs_x = np.arange(0, epochs, 10)
    
    # 1. Validation Accuracy
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs_x, std_hist['val_acc'], label='Standard MLP', color='red', alpha=0.7)
    ax1.plot(epochs_x, topo_hist['val_acc'], label='Topo-Assisted MLP', color='blue', linewidth=2)
    ax1.set_title("Validation Accuracy (Grokking Phase Transition)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs_x, std_hist['val_loss'], label='Standard MLP', color='red', alpha=0.7)
    ax2.plot(epochs_x, topo_hist['val_loss'], label='Topo-Assisted MLP', color='blue', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title("Validation Loss (Log Scale)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cross Entropy Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA of Standard Embeddings (Pull back to CPU for sklearn)
    pca = PCA(n_components=2)
    std_emb = std_model.E_B.weight.detach().cpu().numpy()
    std_pca = pca.fit_transform(std_emb)
    
    ax3 = plt.subplot(2, 2, 3)
    sc3 = ax3.scatter(std_pca[:, 0], std_pca[:, 1], c=np.arange(59), cmap='twilight_shifted', s=50)
    ax3.set_title("Standard MLP: E_B Embeddings (PCA)")
    plt.colorbar(sc3, ax=ax3, label="Token Identity (0-58)")
    
    # 4. PCA of Topo-Assisted Embeddings (Pull back to CPU for sklearn)
    topo_emb = topo_model.E_B.weight.detach().cpu().numpy()
    topo_pca = pca.fit_transform(topo_emb)
    
    ax4 = plt.subplot(2, 2, 4)
    sc4 = ax4.scatter(topo_pca[:, 0], topo_pca[:, 1], c=np.arange(59), cmap='twilight_shifted', s=50)
    ax4.set_title("Topo-Assisted MLP: E_B Embeddings (PCA)")
    plt.colorbar(sc4, ax=ax4, label="Token Identity (0-58)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
