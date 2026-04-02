"""
Reproduction of Lewis & Vrabie (2009):
"Reinforcement Learning and Adaptive Dynamic Programming for Feedback Control"

Progression:
  Stage 1 – Offline Riccati (needs A, B; off-line)
  Stage 2 – Policy Iteration / Value Iteration (Hewer / Lancaster-Rodman; needs A, B; online)
  Stage 3 – TD + VFA single Critic (needs B; online)
  Stage 4 – Actor-Critic dual NN (needs B; online)
  Stage 5 – Q-Learning (needs neither A nor B; online)
  Stage 6 – Continuous-Time Policy Iteration (needs B; online)
"""

import numpy as np
from scipy.linalg import solve_discrete_are, solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM DEFINITION  (DT:  x_{k+1} = A x_k + B u_k)
# ─────────────────────────────────────────────────────────────────────────────
A = np.array([[1.0, 0.1],
              [0.0, 1.0]])          # double-integrator (sampled, T=0.1 s)
B = np.array([[0.005],
              [0.1]])
Q = np.eye(2)                       # state penalty
R = np.array([[0.1]])               # control penalty
n, m = A.shape[0], B.shape[1]

# initial condition for simulations
x0 = np.array([5.0, -3.0])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def simulate(K, steps=200, x_init=None):
    """Roll out u = -K x for `steps` steps, return trajectory."""
    x = x_init.copy() if x_init is not None else x0.copy()
    xs, us, costs = [x.copy()], [], []
    for _ in range(steps):
        u = -K @ x
        cost = float(x @ Q @ x + u @ R @ u)
        x = A @ x + B @ u
        xs.append(x.copy()); us.append(u.copy()); costs.append(cost)
    return np.array(xs), np.array(us), np.array(costs)

def quad_basis(x):
    """Upper-triangular quadratic basis: n(n+1)/2 terms."""
    feats = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            feats.append(x[i] * x[j])
    return np.array(feats)

def quad_basis_aug(x, u):
    """Quadratic basis for joint (x, u) vector – for Q-learning."""
    z = np.concatenate([x, u])
    return quad_basis(z)

def gradient_quad_basis(x):
    """∂(quad_basis)/∂x  shape (L, n)."""
    L = n * (n + 1) // 2
    G = np.zeros((L, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                G[idx, i] = 2 * x[i]
            else:
                G[idx, i] += x[j]
                G[idx, j] += x[i]
            idx += 1
    return G

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 – Offline Riccati (scipy DARE)
# ─────────────────────────────────────────────────────────────────────────────
def stage1_riccati():
    P_star = solve_discrete_are(A, B, Q, R)
    K_star = np.linalg.solve(R + B.T @ P_star @ B, B.T @ P_star @ A)
    return P_star, K_star

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 – Policy Iteration (Hewer) & Value Iteration (Lancaster-Rodman)
# ─────────────────────────────────────────────────────────────────────────────
def stage2_policy_iteration(K_init, max_iter=50, tol=1e-8):
    """Needs A and B explicitly."""
    K = K_init.copy()
    Ps = []
    for _ in range(max_iter):
        Ac = A - B @ K
        # Lyapunov equation: Ac^T P Ac - P + Q + K^T R K = 0
        # => solve  Ac^T P Ac - P = -(Q + K^T R K)
        rhs = Q + K.T @ R @ K
        # vec formulation: (Ac^T ⊗ Ac^T - I) vec(P) = -vec(rhs)
        Av = np.kron(Ac.T, Ac.T) - np.eye(n * n)
        P = np.linalg.solve(Av, -rhs.flatten()).reshape(n, n)
        K_new = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Ps.append(P.copy())
        if np.linalg.norm(K_new - K) < tol:
            K = K_new; break
        K = K_new
    return K, P, Ps

def stage2_value_iteration(K_init, max_iter=200, tol=1e-8):
    """Needs A and B explicitly."""
    K = K_init.copy()
    P = np.eye(n)
    Ps = []
    for _ in range(max_iter):
        Ac = A - B @ K
        P_new = Ac.T @ P @ Ac + Q + K.T @ R @ K   # Lyapunov recursion
        K_new = np.linalg.solve(R + B.T @ P_new @ B, B.T @ P_new @ A)
        Ps.append(P_new.copy())
        if np.linalg.norm(P_new - P) < tol:
            P = P_new; K = K_new; break
        P = P_new; K = K_new
    return K, P, Ps

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 – Online PI with TD error + VFA (single Critic, needs B)
# ─────────────────────────────────────────────────────────────────────────────
def stage3_online_pi_vfa(K_init, n_pi_steps=8, n_rl_steps=600, tol=1e-6, gamma=1.0):
    """
    At each PI iteration:
      - Fix policy K_j
      - Collect (x_k, x_{k+1}, r_k) along trajectory
      - Use RLS on TD equation:  w^T (phi(x_k) - gamma*phi(x_{k+1})) = r_k
      - Update K from critic gradient (needs B)
    """
    K = K_init.copy()
    L = n * (n + 1) // 2     # basis dimension
    Ps_approx = []

    for pi_step in range(n_pi_steps):
        # collect data with current K + small exploration noise
        X_k, X_k1, Rs = [], [], []
        x = x0.copy() + np.random.randn(n) * 0.5
        for _ in range(n_rl_steps):
            u = -K @ x + np.random.randn(m) * 0.05   # probing noise
            r = float(x @ Q @ x + u @ R @ u)
            xn = A @ x + B @ u
            X_k.append(x.copy()); X_k1.append(xn.copy()); Rs.append(r)
            x = xn

        # build regression matrix: Phi = [phi(x_k) - gamma*phi(x_{k+1})]
        Phi = np.array([quad_basis(xk) - gamma * quad_basis(xk1)
                        for xk, xk1 in zip(X_k, X_k1)])
        R_vec = np.array(Rs)

        # Least-squares: w = argmin ||Phi w - R_vec||^2
        w, *_ = np.linalg.lstsq(Phi, R_vec, rcond=None)

        # Recover P from w (upper-triangular packing)
        P_approx = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    P_approx[i, j] = w[idx]
                else:
                    P_approx[i, j] = w[idx] / 2
                    P_approx[j, i] = w[idx] / 2
                idx += 1
        Ps_approx.append(P_approx.copy())

        # policy improvement  K_{j+1} = (R + B^T P B)^{-1} B^T P A  (needs B)
        K = np.linalg.solve(R + B.T @ P_approx @ B, B.T @ P_approx @ A)

    return K, P_approx, Ps_approx

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 – Actor-Critic Dual NN (needs B, not A)
# ─────────────────────────────────────────────────────────────────────────────
def stage4_actor_critic(K_init, n_pi_steps=10, n_rl_steps=800,
                         alpha=0.01, beta=0.005, gamma=1.0):
    """
    Critic: w s.t.  w^T phi(x_k) ≈ V(x_k)   [updated via gradient descent]
    Actor:  U s.t.  u_k = U^T sigma(x_k)     [linear actor: sigma=x, U=-K^T]
            Updated via gradient of critic w.r.t. u (needs B=g(x))
    """
    K = K_init.copy()
    L = n * (n + 1) // 2
    w = np.zeros(L)
    # actor parametrised as u = U^T x  (linear)
    U = -K.T    # shape (n, m)

    Ps_approx, Ks = [], []

    for pi_step in range(n_pi_steps):
        # ── Critic update (RLS-style gradient descent) ──
        x = x0.copy() + np.random.randn(n) * 0.5
        w_c = w.copy()
        for _ in range(n_rl_steps):
            u = U.T @ x + np.random.randn(m) * 0.05
            r = float(x @ Q @ x + u @ R @ u)
            xn = A @ x + B @ u
            phi_k  = quad_basis(x)
            phi_k1 = quad_basis(xn)
            td_err = r + gamma * (w_c @ phi_k1) - (w_c @ phi_k)
            Fk = phi_k - gamma * phi_k1
            w_c = w_c + alpha * Fk * td_err
            x = xn
        w = w_c

        # Recover P
        P_approx = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    P_approx[i, j] = w[idx]
                else:
                    P_approx[i, j] = w[idx] / 2
                    P_approx[j, i] = w[idx] / 2
                idx += 1
        Ps_approx.append(P_approx.copy())

        # ── Actor update via gradient descent (needs B) ──
        x = x0.copy() + np.random.randn(n) * 0.5
        for _ in range(n_rl_steps):
            u_curr = U.T @ x
            xn = A @ x + B @ u_curr
            # ∂V/∂u = g(x)^T ∇_x V  = B^T * (∂phi/∂x)^T w
            grad_phi = gradient_quad_basis(xn)   # (L, n)
            dV_dx = grad_phi.T @ w               # (n,)
            dV_du = B.T @ dV_dx                  # (m,)
            # full gradient w.r.t. U:  ∂/∂U [u^T R u + γ dV_du^T u]
            # u = U^T x  =>  ∂u/∂U = x ⊗ I_m
            grad_U = x.reshape(-1, 1) @ (2 * R @ u_curr + gamma * dV_du).reshape(1, -1)
            U = U - beta * grad_U
            x = xn

        K = -U.T
        Ks.append(K.copy())

    return K, P_approx, Ps_approx, Ks

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 – Q-Learning (needs neither A nor B)
# ─────────────────────────────────────────────────────────────────────────────
def stage5_q_learning(K_init, n_pi_steps=12, n_rl_steps=1000,
                       gamma=1.0, pe_noise=0.3):
    """
    Critic learns Q(x,u) parametrised as  H^T z_k  with z = quad_basis([x;u])
    Policy update: u* = -H_uu^{-1} H_ux x  (no A or B needed)
    PE probing noise added to satisfy persistence of excitation.
    """
    K = K_init.copy()
    aug_dim = (n + m) * (n + m + 1) // 2
    H_vec = np.zeros(aug_dim)
    Hs = []

    for pi_step in range(n_pi_steps):
        # ── Collect data ──
        Phi_list, R_list = [], []
        x = x0.copy() + np.random.randn(n) * 0.5
        for _ in range(n_rl_steps):
            u = -K @ x + np.random.randn(m) * pe_noise   # PE noise
            r  = float(x @ Q @ x + u @ R @ u)
            xn = A @ x + B @ u
            un = -K @ xn    # next action under current policy
            z_k  = quad_basis_aug(x,  u)
            z_k1 = quad_basis_aug(xn, un)
            Phi_list.append(z_k - gamma * z_k1)
            R_list.append(r)
            x = xn

        Phi = np.array(Phi_list)
        R_vec = np.array(R_list)
        H_vec, *_ = np.linalg.lstsq(Phi, R_vec, rcond=None)
        Hs.append(H_vec.copy())

        # ── Reconstruct H matrix ──
        aug = n + m
        H_mat = np.zeros((aug, aug))
        idx = 0
        for i in range(aug):
            for j in range(i, aug):
                if i == j:
                    H_mat[i, j] = H_vec[idx]
                else:
                    H_mat[i, j] = H_vec[idx] / 2
                    H_mat[j, i] = H_vec[idx] / 2
                idx += 1

        H_xx = H_mat[:n, :n]
        H_xu = H_mat[:n, n:]
        H_ux = H_mat[n:, :n]
        H_uu = H_mat[n:, n:]

        # policy improvement: u* = -H_uu^{-1} H_ux x  (no A, no B!)
        try:
            K = np.linalg.solve(H_uu, H_ux)
        except np.linalg.LinAlgError:
            pass   # keep old K if singular

    return K, H_vec, H_mat, Hs

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 – Continuous-Time Policy Iteration (needs B, not A)
# ─────────────────────────────────────────────────────────────────────────────
def stage6_ct_policy_iteration(K_init, n_pi_steps=10,
                                 T_interval=0.5, n_samples=300,
                                 dt=0.01, tol=1e-6):
    """
    CT system:  ẋ = f(x) + g(x)u  with f=Ax, g=B  (only B needed for update)
    Interval Bellman eq:
        V(x(t)) = ∫_{t}^{t+T} r(x,u) dτ + V(x(t+T))
    Solve for w via LS on:
        w^T [phi(x(t)) - phi(x(t+T))] = ∫_{t}^{t+T} r dτ
    Policy:  u = -½ R^{-1} B^T ∇V(x)  = -½ R^{-1} B^T (∂phi/∂x)^T w
    """
    # CT system matrices (use same A, B for consistency)
    A_ct = np.array([[0.0, 1.0],
                     [0.0, 0.0]])    # continuous-time double integrator
    B_ct = np.array([[0.0],
                     [1.0]])
    Q_ct = np.eye(n)
    R_ct = np.array([[0.1]])

    K = K_init.copy()
    L = n * (n + 1) // 2
    Ps_approx = []

    def simulate_ct(x_start, K_ct, duration, dt_sim):
        """Euler integration of ẋ = A_ct x - B_ct K x, returns (xs, int_cost)."""
        x = x_start.copy()
        xs = [x.copy()]
        int_cost = 0.0
        steps = int(duration / dt_sim)
        for _ in range(steps):
            u = -K_ct @ x + np.random.randn(m) * 0.02
            int_cost += float(x @ Q_ct @ x + u @ R_ct @ u) * dt_sim
            x = x + (A_ct @ x + B_ct @ u) * dt_sim
            xs.append(x.copy())
        return np.array(xs), int_cost, x

    for pi_step in range(n_pi_steps):
        Phi_list, IntR_list = [], []
        x = x0.copy() + np.random.randn(n) * 0.3

        for _ in range(n_samples):
            xs_seg, int_r, x_end = simulate_ct(x, K, T_interval, dt)
            phi_start = quad_basis(x)
            phi_end   = quad_basis(x_end)
            Phi_list.append(phi_start - phi_end)
            IntR_list.append(int_r)
            x = x_end + np.random.randn(n) * 0.1   # reset with variation

        Phi = np.array(Phi_list)
        IntR = np.array(IntR_list)
        w, *_ = np.linalg.lstsq(Phi, IntR, rcond=None)

        P_approx = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    P_approx[i, j] = w[idx]
                else:
                    P_approx[i, j] = w[idx] / 2
                    P_approx[j, i] = w[idx] / 2
                idx += 1
        Ps_approx.append(P_approx.copy())

        # CT policy update: u = -½ R^{-1} B^T ∇V  (needs B_ct)
        # For linear system V=x^T P x: ∇V = 2Px  =>  K = R^{-1} B^T P
        K = np.linalg.solve(R_ct, B_ct.T @ P_approx)

    return K, P_approx, Ps_approx, A_ct, B_ct, Q_ct, R_ct


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL STAGES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Lewis & Vrabie (2009) – Reproduction")
print("=" * 60)

# Ground truth
P_star, K_star = stage1_riccati()
print(f"\n[Stage 1] Riccati K* = {K_star.flatten()}")
print(f"          P* =\n{P_star}")

# Stabilizing init
K_init = np.array([[2.0, 1.5]])   # hand-tuned stabilizing gain

print("\n[Stage 2] Policy Iteration (Hewer)…")
K_pi, P_pi, Ps_pi = stage2_policy_iteration(K_init)
print(f"  K_PI = {K_pi.flatten()}  (err = {np.linalg.norm(K_pi-K_star):.2e})")

print("\n[Stage 2] Value Iteration (Lancaster-Rodman)…")
K_vi, P_vi, Ps_vi = stage2_value_iteration(K_init)
print(f"  K_VI = {K_vi.flatten()}  (err = {np.linalg.norm(K_vi-K_star):.2e})")

print("\n[Stage 3] Online PI + TD/VFA (single Critic, needs B)…")
K_s3, P_s3, Ps_s3 = stage3_online_pi_vfa(K_init)
print(f"  K_s3 = {K_s3.flatten()}  (err = {np.linalg.norm(K_s3-K_star):.2e})")

print("\n[Stage 4] Actor-Critic dual NN (needs B, not A)…")
K_s4, P_s4, Ps_s4, Ks_s4 = stage4_actor_critic(K_init)
print(f"  K_s4 = {K_s4.flatten()}  (err = {np.linalg.norm(K_s4-K_star):.2e})")

print("\n[Stage 5] Q-Learning (needs neither A nor B)…")
K_s5, H_vec, H_mat, Hs_s5 = stage5_q_learning(K_init)
print(f"  K_s5 = {K_s5.flatten()}  (err = {np.linalg.norm(K_s5-K_star):.2e})")

print("\n[Stage 6] Continuous-Time Policy Iteration…")
K_s6, P_s6, Ps_s6, A_ct, B_ct, Q_ct, R_ct = stage6_ct_policy_iteration(
    np.array([[1.0, 2.0]]))
P_star_ct = solve_discrete_are(
    np.eye(n) + A_ct * 0.01,
    B_ct * 0.01,
    Q_ct * 0.01, R_ct)   # approximate CT Riccati via fine DT
print(f"  K_s6 = {K_s6.flatten()}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORIES  (DT stages 1-5)
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "Stage 1\nRiccati (offline)":       K_star,
    "Stage 2\nPolicy Iteration":         K_pi,
    "Stage 2\nValue Iteration":          K_vi,
    "Stage 3\nTD+VFA (single critic)":  K_s3,
    "Stage 4\nActor-Critic":            K_s4,
    "Stage 5\nQ-Learning":              K_s5,
}

trajectories = {}
for label, K in results.items():
    xs, us, costs = simulate(K)
    trajectories[label] = (xs, us, costs)

# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
def P_err(Ps_list):
    return [np.linalg.norm(P - P_star, 'fro') for P in Ps_list]

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
colors = {
    "Stage 1\nRiccati (offline)":      "#E63946",
    "Stage 2\nPolicy Iteration":        "#2A9D8F",
    "Stage 2\nValue Iteration":         "#457B9D",
    "Stage 3\nTD+VFA (single critic)": "#E9C46A",
    "Stage 4\nActor-Critic":           "#F4A261",
    "Stage 5\nQ-Learning":             "#A8DADC",
}

fig = plt.figure(figsize=(20, 22), facecolor='#0D1117')
fig.suptitle("Lewis & Vrabie (2009)  ·  Reinforcement Learning & ADP for Feedback Control",
             fontsize=16, color='#E6EDF3', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.95, bottom=0.04)

ax_style = dict(facecolor='#161B22', labelcolor='#8B949E',
                tickcolor='#8B949E', spinecolor='#30363D')

def style_ax(ax, title):
    ax.set_facecolor(ax_style['facecolor'])
    ax.tick_params(colors=ax_style['tickcolor'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(ax_style['spinecolor'])
    ax.set_title(title, color='#E6EDF3', fontsize=9, pad=6, fontweight='bold')
    ax.xaxis.label.set_color(ax_style['labelcolor'])
    ax.yaxis.label.set_color(ax_style['labelcolor'])
    ax.grid(True, color='#21262D', linewidth=0.5, linestyle='--')

# ── Row 0: state x1 trajectories ──────────────────────────────────────────
ax_x1 = fig.add_subplot(gs[0, :])
style_ax(ax_x1, "State x₁ Trajectories  (all stages)")
for label, (xs, us, costs) in trajectories.items():
    short = label.replace('\n', ' ')
    ax_x1.plot(xs[:, 0], color=colors[label], linewidth=1.8,
               label=short, alpha=0.9)
ax_x1.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')
ax_x1.set_xlabel("Time step k"); ax_x1.set_ylabel("x₁")
ax_x1.legend(loc='upper right', fontsize=7, facecolor='#161B22',
             edgecolor='#30363D', labelcolor='#E6EDF3', ncol=3)

# ── Row 1 col 0: x2 trajectory ────────────────────────────────────────────
ax_x2 = fig.add_subplot(gs[1, 0])
style_ax(ax_x2, "State x₂ Trajectories")
for label, (xs, us, costs) in trajectories.items():
    ax_x2.plot(xs[:, 1], color=colors[label], linewidth=1.5, alpha=0.85)
ax_x2.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')
ax_x2.set_xlabel("Time step k"); ax_x2.set_ylabel("x₂")

# ── Row 1 col 1: control input ────────────────────────────────────────────
ax_u = fig.add_subplot(gs[1, 1])
style_ax(ax_u, "Control Input u")
for label, (xs, us, costs) in trajectories.items():
    ax_u.plot(us[:, 0], color=colors[label], linewidth=1.5, alpha=0.85)
ax_u.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')
ax_u.set_xlabel("Time step k"); ax_u.set_ylabel("u")

# ── Row 1 col 2: cumulative cost ──────────────────────────────────────────
ax_cost = fig.add_subplot(gs[1, 2])
style_ax(ax_cost, "Cumulative Cost")
for label, (xs, us, costs) in trajectories.items():
    ax_cost.plot(np.cumsum(costs), color=colors[label], linewidth=1.5, alpha=0.85)
ax_cost.set_xlabel("Time step k"); ax_cost.set_ylabel("Σ r(x,u)")

# ── Row 2: PI & VI convergence ────────────────────────────────────────────
ax_pi = fig.add_subplot(gs[2, 0])
style_ax(ax_pi, "Stage 2 – Policy Iteration Convergence")
errs_pi = P_err(Ps_pi)
ax_pi.semilogy(errs_pi, color='#2A9D8F', linewidth=2, marker='o', ms=5)
ax_pi.set_xlabel("PI iteration"); ax_pi.set_ylabel("‖P_j − P*‖_F")

ax_vi = fig.add_subplot(gs[2, 1])
style_ax(ax_vi, "Stage 2 – Value Iteration Convergence")
errs_vi = P_err(Ps_vi)
ax_vi.semilogy(errs_vi, color='#457B9D', linewidth=2, marker='s', ms=4)
ax_vi.set_xlabel("VI iteration"); ax_vi.set_ylabel("‖P_j − P*‖_F")

# ── Row 2 col 2: Stage 3 VFA convergence ─────────────────────────────────
ax_s3 = fig.add_subplot(gs[2, 2])
style_ax(ax_s3, "Stage 3 – Online VFA Convergence")
errs_s3 = P_err(Ps_s3)
ax_s3.semilogy(errs_s3, color='#E9C46A', linewidth=2, marker='^', ms=5)
ax_s3.set_xlabel("PI (outer) iteration"); ax_s3.set_ylabel("‖P̂ − P*‖_F")

# ── Row 3 col 0: Stage 4 Actor-Critic gain convergence ────────────────────
ax_s4 = fig.add_subplot(gs[3, 0])
style_ax(ax_s4, "Stage 4 – Actor-Critic Gain Convergence")
K_errs_s4 = [np.linalg.norm(Kj - K_star) for Kj in Ks_s4]
ax_s4.semilogy(K_errs_s4, color='#F4A261', linewidth=2, marker='D', ms=5)
ax_s4.set_xlabel("PI iteration"); ax_s4.set_ylabel("‖K_j − K*‖")

# ── Row 3 col 1: Q-Learning H_uu^{-1} H_ux convergence ───────────────────
ax_s5 = fig.add_subplot(gs[3, 1])
style_ax(ax_s5, "Stage 5 – Q-Learning Gain Convergence")
# recover K at each step
QL_K_errs = []
for h in Hs_s5:
    aug = n + m
    H_m = np.zeros((aug, aug))
    idx = 0
    for i in range(aug):
        for j in range(i, aug):
            if i == j: H_m[i,j] = h[idx]
            else:      H_m[i,j] = H_m[j,i] = h[idx]/2
            idx += 1
    try:
        K_tmp = np.linalg.solve(H_m[n:,n:], H_m[n:,:n])
        QL_K_errs.append(np.linalg.norm(K_tmp - K_star))
    except:
        QL_K_errs.append(np.nan)
ax_s5.semilogy(QL_K_errs, color='#A8DADC', linewidth=2, marker='v', ms=5)
ax_s5.set_xlabel("Q-learning iteration"); ax_s5.set_ylabel("‖K_j − K*‖")

# ── Row 3 col 2: CT PI convergence ───────────────────────────────────────
ax_s6 = fig.add_subplot(gs[3, 2])
style_ax(ax_s6, "Stage 6 – CT Policy Iteration Convergence")
# CT Riccati reference via fine DT
dt_fine = 1e-3
P_star_ct = solve_discrete_are(
    np.eye(n) + A_ct * dt_fine,
    B_ct * dt_fine,
    Q_ct * dt_fine, R_ct) / dt_fine
errs_s6 = [np.linalg.norm(P - P_star_ct, 'fro') for P in Ps_s6]
ax_s6.semilogy(errs_s6, color='#C77DFF', linewidth=2, marker='p', ms=6)
ax_s6.set_xlabel("CT PI iteration"); ax_s6.set_ylabel("‖P̂ − P*_ct‖_F")

plt.savefig('/mnt/user-data/outputs/lewis2009_reproduction.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1117')
print("\n✓ Figure saved to outputs/lewis2009_reproduction.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Stage':<35} {'K error':>12} {'Needs A':>9} {'Needs B':>9} {'Online':>8}")
print("-" * 60)
rows = [
    ("Stage 1: Riccati (offline)",          np.linalg.norm(K_star-K_star), True,  True,  False),
    ("Stage 2: Policy Iteration (Hewer)",   np.linalg.norm(K_pi-K_star),   True,  True,  True),
    ("Stage 2: Value Iteration (L-R)",      np.linalg.norm(K_vi-K_star),   True,  True,  True),
    ("Stage 3: TD+VFA single Critic",       np.linalg.norm(K_s3-K_star),   False, True,  True),
    ("Stage 4: Actor-Critic dual NN",       np.linalg.norm(K_s4-K_star),   False, True,  True),
    ("Stage 5: Q-Learning",                 np.linalg.norm(K_s5-K_star),   False, False, True),
]
for name, err, nA, nB, online in rows:
    print(f"{name:<35} {err:>12.2e} {'Yes' if nA else 'No':>9} {'Yes' if nB else 'No':>9} {'Yes' if online else 'No':>8}")
print("=" * 60)
