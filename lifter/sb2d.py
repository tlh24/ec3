"""
sb2d.py — Schrödinger Bridge toy test: 8-moons ↔ S-curve (2-D)

Algorithm (De Bortoli et al. 2021, arxiv:2106.01357), discrete-time IPFP:

  Integration step (same formula for both F and B):
      x_{k+1} = x_k + γ · net(x_k, k/(N+1)) + √(2γ) · Z,   Z ~ N(0,I)
      γ = σ₀ / (N+1)

  Alternating phases:
    Phase B  (F fixed):  simulate F from data_0;  for random k,
        target_B = x_{k+1} + F(x_k, t_k) − F(x_{k+1}, t_k)
        loss_B   = ||B(x_{k+1}, t_{k+1}) − target_B||²

    Phase F  (B fixed):  simulate B from data_1;  for random k,
        target_F = x_k + B(x_{k+1}, t_{k+1}) − B(x_k, t_{k+1})
        loss_F   = ||F(x_k, t_k) − target_F||²

  F(x_k, t_k) is cached from simulation; one fresh net call per loss step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_s_curve

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_STEPS    = 20        # discrete integration steps  (x_0 … x_N)
SIGMA_0    = 1.0       # total diffusion strength
GAMMA      = SIGMA_0 / (N_STEPS + 1)
NOISE_STD  = math.sqrt(2 * GAMMA)

BATCH      = 512
N_ALT      = 500       # number of alternations
STEPS_PER  = 100       # gradient steps per phase per alternation
LR         = 3e-4

HIDDEN     = 256
N_LAYERS   = 3         # hidden layers between in_proj and out_proj
T_EMB_DIM  = 32
SMALL_INIT_STD = 0.01  # keeps initial drift near zero → stable early trajectories

PLOT_EVERY = 50
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Time embedding ─────────────────────────────────────────────────────────────
def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t : [B] in [0, 1]  →  [B, dim]"""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]          # [B, half]
    return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


# ── Network ────────────────────────────────────────────────────────────────────
class ScoreNet(nn.Module):
    """
    MLP that takes (x ∈ R^D, t ∈ [0,1]) and returns a velocity ∈ R^D.
    Time is injected via sinusoidal embedding added to the first hidden layer.
    Residual connections on hidden layers for stable gradient flow.
    Weights initialised to near-zero so initial drift ≈ 0.
    """
    def __init__(self, data_dim: int = 2):
        super().__init__()
        self.t_proj   = nn.Linear(T_EMB_DIM, HIDDEN)
        self.in_proj  = nn.Linear(data_dim,  HIDDEN)
        self.layers   = nn.ModuleList(
            [nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS)]
        )
        self.out_proj = nn.Linear(HIDDEN, data_dim)
        self._small_init()

    def _small_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=SMALL_INIT_STD)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x : [B, D],  t : [B]  →  [B, D]"""
        t_emb = sinusoidal_emb(t, T_EMB_DIM)              # [B, T_EMB_DIM]
        h = F_nn.silu(self.in_proj(x) + self.t_proj(t_emb))
        for layer in self.layers:
            h = F_nn.silu(layer(h) + h)                   # residual
        return self.out_proj(h)


# ── Data samplers ──────────────────────────────────────────────────────────────
def sample_moons(n: int) -> torch.Tensor:
    X, _ = make_moons(n_samples=n, noise=0.05)
    return torch.tensor(X, dtype=torch.float32, device=DEVICE)


def sample_scurve(n: int) -> torch.Tensor:
    """Project 3-D S-curve onto (x, z) plane."""
    X, _ = make_s_curve(n_samples=n, noise=0.1)
    return torch.tensor(X[:, [0, 2]], dtype=torch.float32, device=DEVICE)


# ── SDE simulation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def simulate(
    net: ScoreNet,
    x0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate SDE for N_STEPS starting from x0.

    Returns
    -------
    traj : [B, N_STEPS+1, D]  — trajectory positions
    vals : [B, N_STEPS+1, D]  — cached net evaluations
                                 vals[:, k] = net(traj[:, k], k/(N_STEPS+1))
    """
    B, D = x0.shape
    traj = x0.new_zeros(B, N_STEPS + 1, D)
    vals = x0.new_zeros(B, N_STEPS + 1, D)
    traj[:, 0] = x0
    for k in range(N_STEPS):
        t = x0.new_full((B,), k / (N_STEPS + 1))
        v = net(traj[:, k], t)
        vals[:, k] = v
        noise = torch.randn_like(traj[:, k]) * NOISE_STD
        traj[:, k + 1] = traj[:, k] + GAMMA * v + noise
    # Cache eval at the final position (needed when k = N_STEPS-1 is selected)
    t_end = x0.new_full((B,), N_STEPS / (N_STEPS + 1))
    vals[:, N_STEPS] = net(traj[:, N_STEPS], t_end)
    return traj, vals


# ── Training phases ────────────────────────────────────────────────────────────
def phase_b(F_net: ScoreNet, B_net: ScoreNet, opt_b: torch.optim.Optimizer) -> float:
    """
    Fix F.  Simulate from data_0.  Train B.

    target_B = x_{k+1} + F(x_k, t_k) − F(x_{k+1}, t_k)
    loss_B   = ||B(x_{k+1}, t_{k+1}) − target_B||²
    """
    total = 0.0
    for _ in range(STEPS_PER):
        x0 = sample_moons(BATCH)
        with torch.no_grad():
            traj, f_vals = simulate(F_net, x0)
            k   = torch.randint(0, N_STEPS, (1,)).item()
            xk  = traj[:, k]
            xk1 = traj[:, k + 1]
            tk  = x0.new_full((BATCH,), k       / (N_STEPS + 1))
            tk1 = x0.new_full((BATCH,), (k + 1) / (N_STEPS + 1))
            fk_xk  = f_vals[:, k]          # F(x_k,  t_k) — cached from simulation
            fk_xk1 = F_net(xk1, tk)        # F(x_{k+1}, t_k) — fresh (time t_k, not t_{k+1})
            target = xk1 + fk_xk - fk_xk1  # [B, D]

        pred = B_net(xk1, tk1)             # B(x_{k+1}, t_{k+1})
        loss = ((pred - target) ** 2).mean()
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()
        total += loss.item()
    return total / STEPS_PER


def phase_f(F_net: ScoreNet, B_net: ScoreNet, opt_f: torch.optim.Optimizer) -> float:
    """
    Fix B.  Simulate from data_1.  Train F.

    target_F = x_k + B(x_{k+1}, t_{k+1}) − B(x_k, t_{k+1})
    loss_F   = ||F(x_k, t_k) − target_F||²
    """
    total = 0.0
    for _ in range(STEPS_PER):
        x1 = sample_scurve(BATCH)
        with torch.no_grad():
            traj, b_vals = simulate(B_net, x1)
            k    = torch.randint(0, N_STEPS, (1,)).item()
            xk   = traj[:, k]
            xk1  = traj[:, k + 1]
            tk   = x1.new_full((BATCH,), k       / (N_STEPS + 1))
            tk1  = x1.new_full((BATCH,), (k + 1) / (N_STEPS + 1))
            bk1_xk1 = b_vals[:, k + 1]        # B(x_{k+1}, t_{k+1}) — cached
            bk1_xk  = B_net(xk, tk1)           # B(x_k,    t_{k+1}) — fresh (pos x_k, not x_{k+1})
            target = xk + bk1_xk1 - bk1_xk    # [B, D]

        pred = F_net(xk, tk)                   # F(x_k, t_k)
        loss = ((pred - target) ** 2).mean()
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()
        total += loss.item()
    return total / STEPS_PER


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot(F_net: ScoreNet, B_net: ScoreNet, alt: int) -> None:
    N_VIS = 1000
    x0 = sample_moons(N_VIS)
    x1 = sample_scurve(N_VIS)
    with torch.no_grad():
        traj_f, _ = simulate(F_net, x0)
        traj_b, _ = simulate(B_net, x1)
    xf = traj_f[:, -1].cpu().numpy()
    xb = traj_b[:, -1].cpu().numpy()
    x0n = x0.cpu().numpy()
    x1n = x1.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].scatter(*x0n.T, s=3, alpha=0.5)
    axes[0].set_title('data_0  (moons)')

    axes[1].scatter(*x1n.T, s=3, alpha=0.5, color='C1')
    axes[1].set_title('data_1  (s-curve)')

    axes[2].scatter(*xf.T,  s=3, alpha=0.5, color='C2', label='F(data_0)')
    axes[2].scatter(*x1n.T, s=3, alpha=0.2, color='C1', label='data_1')
    axes[2].set_title(f'Forward transport  alt={alt}')
    axes[2].legend(markerscale=3, fontsize=7)

    axes[3].scatter(*xb.T,  s=3, alpha=0.5, color='C3', label='B(data_1)')
    axes[3].scatter(*x0n.T, s=3, alpha=0.2, color='C0', label='data_0')
    axes[3].set_title(f'Backward transport  alt={alt}')
    axes[3].legend(markerscale=3, fontsize=7)

    plt.tight_layout()
    plt.savefig(f'sb2d_{alt:04d}.png', dpi=100)
    plt.close()
    print(f'  → saved sb2d_{alt:04d}.png')


def plot_trajectories(F_net: ScoreNet, alt: int, n_traj: int = 30) -> None:
    """Show a handful of trajectory paths to diagnose drift behaviour."""
    x0 = sample_moons(n_traj)
    with torch.no_grad():
        traj, _ = simulate(F_net, x0)
    traj_np = traj.cpu().numpy()   # [n_traj, N_STEPS+1, 2]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_traj):
        ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], '-', alpha=0.4, linewidth=0.8)
        ax.scatter(traj_np[i,  0, 0], traj_np[i,  0, 1], c='blue',  s=15, zorder=3)
        ax.scatter(traj_np[i, -1, 0], traj_np[i, -1, 1], c='red',   s=15, zorder=3)
    ax.set_title(f'F trajectories (blue=start, red=end)  alt={alt}')
    plt.tight_layout()
    plt.savefig(f'sb2d_traj_{alt:04d}.png', dpi=100)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device : {DEVICE}")
    print(f"γ={GAMMA:.5f}   √(2γ)={NOISE_STD:.4f}   N={N_STEPS}   σ₀={SIGMA_0}")

    F_net = ScoreNet().to(DEVICE)
    B_net = ScoreNet().to(DEVICE)
    opt_f = torch.optim.Adam(F_net.parameters(), lr=LR)
    opt_b = torch.optim.Adam(B_net.parameters(), lr=LR)

    n_params = sum(p.numel() for p in F_net.parameters())
    print(f"Params per net: {n_params:,}")

    for alt in range(N_ALT):
        lb = phase_b(F_net, B_net, opt_b)
        lf = phase_f(F_net, B_net, opt_f)
        if alt % 10 == 0:
            print(f"alt {alt:4d}  loss_B={lb:.5f}  loss_F={lf:.5f}")
        if alt % PLOT_EVERY == 0 or alt == N_ALT - 1:
            plot(F_net, B_net, alt)
            plot_trajectories(F_net, alt)


if __name__ == '__main__':
    main()
