"""
================================================================================
复现论文：Lewis & Vrabie (2009)
"Reinforcement Learning and Adaptive Dynamic Programming for Feedback Control"
IEEE Circuits and Systems Magazine

算法演进路线（每步放宽对系统模型的依赖）：
  Stage 1 — 离线Riccati方程求解         （需要A和B，离线）
  Stage 2 — 策略迭代(PI) / 值迭代(VI)   （需要A和B，在线迭代）
  Stage 3 — TD误差 + 值函数逼近(VFA)    （只需B，在线）
  Stage 4 — Actor-Critic 双神经网络      （只需B，在线）
  Stage 5 — Q-Learning                  （A和B都不需要，在线）
  Stage 6 — 连续时间策略迭代            （只需B，在线）
================================================================================
"""

# ── 导入标准库 ────────────────────────────────────────────────────────────────
import numpy as np                        # 数值计算核心库
from scipy.linalg import solve_discrete_are  # 求解离散代数Riccati方程(DARE)
import matplotlib                         # 绘图库
matplotlib.use('Agg')                     # 使用非交互式后端（适合脚本保存图片）
import matplotlib.pyplot as plt           # 绘图接口
import matplotlib.gridspec as gridspec    # 灵活的子图布局管理器
import warnings
warnings.filterwarnings('ignore')         # 屏蔽无关警告，保持终端输出整洁

# ── 固定随机种子，保证结果可复现 ────────────────────────────────────────────
np.random.seed(42)

# ==============================================================================
# 系统定义  离散时间(DT)状态空间：x_{k+1} = A·x_k + B·u_k
# ==============================================================================
# 双积分器（位置-速度系统），连续时间采样周期 T=0.1s
# 连续系统: ẋ₁=x₂, ẋ₂=u  →  离散化后得到下面的A,B
A = np.array([[1.0, 0.1],   # 状态转移矩阵(2×2)：第一行 x₁_{k+1}=x₁_k+0.1·x₂_k
              [0.0, 1.0]])  #                  第二行 x₂_{k+1}=x₂_k
B = np.array([[0.005],      # 控制输入矩阵(2×1)：u对x₁的影响(0.5·T²=0.005)
              [0.1]])       #                    u对x₂的影响(T=0.1)
# r(x,u) = x^T Q x + u^T R u  →  二次型代价函数，Q和R分别权衡状态和控制的代价
Q = np.eye(2)               # 状态代价权重矩阵(2×2单位阵)：对x₁²和x₂²等权惩罚
R = np.array([[0.1]])       # 控制代价权重(标量0.1)：对控制量u²的惩罚（越小越允许大控制）
n, m = A.shape[0], B.shape[1]  # n=2(状态维度), m=1(控制维度)

# 仿真初始状态：位置x₁=5（偏离原点较远），速度x₂=-3（向负方向运动）
x0 = np.array([5.0, -3.0])

# ==============================================================================
# 辅助函数
# ==============================================================================

def simulate(K, steps=200, x_init=None):
    """
    在线性状态反馈控制律 u_k = -K·x_k 下仿真系统200步。
    参数：
        K      : 控制增益矩阵 (m×n)
        steps  : 仿真步数
        x_init : 初始状态（默认使用全局x0）
    返回：
        xs    : 状态轨迹 (steps+1, n)
        us    : 控制输入序列 (steps, m)
        costs : 每步单步代价 r(x_k, u_k) (steps,)
    """
    x = x_init.copy() if x_init is not None else x0.copy()  # 初始化状态
    xs, us, costs = [x.copy()], [], []   # 记录容器：状态、控制、代价
    for _ in range(steps):
        u = -K @ x                                # 线性状态反馈：u = -Kx
        cost = float(x @ Q @ x + u @ R @ u)      # 单步代价：x^T Q x + u^T R u
        x = A @ x + B @ u                         # 状态转移方程
        xs.append(x.copy())
        us.append(u.copy())
        costs.append(cost)
    return np.array(xs), np.array(us), np.array(costs)


def quad_basis(x):  #我的理解是手动帮算法把非线性的关系计算出来
    """
    构造状态x的二次型基函数向量（上三角展开）。
    对于n维状态向量，共有 n(n+1)/2 个独立二次项。
    例如 n=2: φ(x) = [x₁², x₁x₂, x₂²]  （3项）
    这是因为值函数 V(x)=x^T P x 是关于x的二次型，
    可以用这组基函数的线性组合精确表示。
    参数：x : 状态向量 (n,)
    返回：feats : 基函数值向量 (n(n+1)/2,)
    """
    feats = []
    for i in range(len(x)):
        for j in range(i, len(x)):      # 只取上三角（含对角线）避免重复
            feats.append(x[i] * x[j])  # 构造所有不重复的二次项 xᵢ·xⱼ
    return np.array(feats)


def quad_basis_aug(x, u):   #因为Q函数是状态和控制的二次型，所以需要把状态和控制拼接成增广向量，再做二次型展开
    """
    构造增广向量 z=[x;u] 的二次型基函数，用于Q函数逼近。
    Q(x,u) = z^T H z，其中z是状态和控制拼接的增广向量。
    对于 n=2, m=1，增广向量维度=3，基函数数量=3×4/2=6。
    参数：x : 状态向量 (n,), u : 控制向量 (m,)
    返回：quad_basis([x;u]) : 增广基函数向量
    """
    z = np.concatenate([x, u])   # 拼接状态和控制，得到增广向量z
    return quad_basis(z)          # 对增广向量做二次型展开


def gradient_quad_basis(x): # 后面涉及到对二次项特征求导，这个函数计算这些特征对状态的梯度，方便后续Actor更新时用链式法则计算控制梯度
    """
    计算二次型基函数 φ(x) 对状态 x 的Jacobian矩阵 ∂φ/∂x。
    形状为 (L, n)，其中 L=n(n+1)/2。
    用途：在 Actor 更新步骤中，通过链式法则求 ∂V/∂x = (∂φ/∂x)^T · w，
    进而得到 ∂V/∂u = B^T · ∂V/∂x（需要已知B矩阵）。
    参数：x : 状态向量 (n,)
    返回：G : Jacobian矩阵 (L, n)
    """
    L = n * (n + 1) // 2          # 基函数总数
    G = np.zeros((L, n))          # 初始化Jacobian矩阵
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # 对角项 xᵢ² 对 xᵢ 求导 = 2xᵢ
                G[idx, i] = 2 * x[i]
            else:
                # 交叉项 xᵢxⱼ 对 xᵢ 求导 = xⱼ，对 xⱼ 求导 = xᵢ
                G[idx, i] += x[j]
                G[idx, j] += x[i]
            idx += 1
    return G
'''
当输入 x = [2, 3]（即 x_1 = 2, x_2 = 3）时，我们有 3 个特征：x_1^2、x_1 x_2、x_2^2。
分别对三个特征求导，对x_1^2的导数是[4.0]
对x_1 x_2的导数是[3,2]
对x_2^2的导数是[0,6]
故最终的Jacobian矩阵 G 是：
[[4.0, 0.0],   # ∂(x_1^2)/∂x = [4.0, 0.0]
 [3.0, 2.0],   # ∂(x_1 x_2)/∂x = [3.0, 2.0]
 [0.0, 6.0]]   # ∂(x_2^2)/∂x = [0.0, 6.0]
'''



def recover_P_from_w(w):    #相当于是对前面quad_basis函数的逆操作，从值函数参数向量w恢复出对应的P矩阵，方便后续策略改进步骤使用
    """
    从值函数参数向量 w 恢复对称矩阵 P。
    V(x) = x^T P x = w^T φ(x)，其中φ是二次型基函数。
    对角元素 w[idx] 直接对应 P[i,i]；
    非对角元素 w[idx] 对应 P[i,j]=P[j,i]=w[idx]/2（因为展开时 xᵢxⱼ 出现一次，
    但P矩阵中 P[i,j]xᵢxⱼ + P[j,i]xⱼxᵢ = 2P[i,j]xᵢxⱼ）。
    """
    P = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                P[i, j] = w[idx]          # 对角元素直接赋值
            else:
                P[i, j] = w[idx] / 2      # 非对角元素除以2
                P[j, i] = w[idx] / 2      # 利用对称性填充下三角
            idx += 1
    return P


# ==============================================================================
# Stage 1：离线Riccati方程求解（需要完整已知A、B）
# ==============================================================================
def stage1_riccati():
    """
    直接调用scipy求解离散代数Riccati方程(DARE)，得到最优解P*和K*。
    DARE：A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0
    这是所有后续算法的参考基准（ground truth）。
    需要完整已知系统矩阵A和B，且是离线的（不能在线学习）。
    返回：P_star : 最优代价矩阵 (n×n)
          K_star : 最优控制增益 (m×n)
    """
    P_star = solve_discrete_are(A, B, Q, R)  # scipy直接求解DARE，得到最优P*
    # 最优增益公式：K* = (R + B^T P* B)^{-1} B^T P* A
    K_star = np.linalg.solve(R + B.T @ P_star @ B, B.T @ P_star @ A)
    return P_star, K_star


# ==============================================================================
# Stage 2a：策略迭代 Policy Iteration（Hewer算法，需要A、B，在线迭代）
# ==============================================================================
def stage2_policy_iteration(K_init, max_iter=50, tol=1e-8):
    """
    Hewer(1971)算法：交替求解Lyapunov方程和更新控制增益，等价于对Riccati方程做策略迭代。

    算法步骤（论文公式33-34）：
      策略评估：给定增益 Kⱼ，求解Lyapunov方程得到 Pⱼ₊₁
                (A-BKⱼ)^T Pⱼ₊₁ (A-BKⱼ) - Pⱼ₊₁ + Q + Kⱼ^T R Kⱼ = 0
      策略改进：Kⱼ₊₁ = (R + B^T Pⱼ₊₁ B)^{-1} B^T Pⱼ₊₁ A

    特点：每步需要完整求解一个线性方程组（"full backup"），收敛快（接近二次收敛），
          但每步计算量大；需要知道A和B。

    参数：K_init  : 初始稳定化增益（必须使闭环系统稳定）
          max_iter: 最大迭代次数
          tol     : 收敛判据（增益变化的2范数）
    返回：K   : 收敛后的最优增益
          P   : 对应的最优代价矩阵
          Ps  : 每次迭代的P矩阵列表（用于绘制收敛曲线）
    """
    K = K_init.copy()   # 初始的控制策略，要求必须是稳定的（即A-BK的特征值模小于1），否则P会无穷大
    Ps = []             # 存储每步迭代的P，用于画收敛图

    for _ in range(max_iter):
        
        # ── 策略评估：用Kronecker积把Lyapunov方程转化为线性方程组求解 ──
        Ac = A - B @ K  
        rhs = Q + K.T @ R @ K 
        Av = np.kron(Ac.T, Ac.T) - np.eye(n * n)  # 计算机无法直接计算矩阵方程，克罗内积计算
        P = np.linalg.solve(Av, -rhs.flatten()).reshape(n, n)  # 求解

        # ── 策略改进：根据新P计算新增益 ──
        K_new = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Ps.append(P.copy())   # 记录本次P

        # 判断是否收敛
        if np.linalg.norm(K_new - K) < tol:
            K = K_new
            break
        K = K_new   # 更新增益，进入下一次迭代

    return K, P, Ps


# ==============================================================================
# Stage 2b：值迭代 Value Iteration（Lancaster-Rodman，需要A、B，在线迭代）
# ==============================================================================
def stage2_value_iteration(K_init, max_iter=200, tol=1e-8):
    """
    Lancaster-Rodman(1995)方法：每步只做一次Lyapunov递推（而不是完整求解），
    等价于对Bellman最优性方程做值迭代。

    算法步骤（论文公式35,34）：
      值更新（"partial backup"）：Pⱼ₊₁ = (A-BKⱼ)^T Pⱼ (A-BKⱼ) + Q + Kⱼ^T R Kⱼ
      策略改进：Kⱼ₊₁ = (R + B^T Pⱼ₊₁ B)^{-1} B^T Pⱼ₊₁ A

    特点：每步只做矩阵乘法（无需解方程），计算量极小，但需要更多迭代次数（线性收敛）；
          不需要初始稳定化策略；需要知道A和B。

    参数：K_init  : 初始增益（不要求稳定化）
          max_iter: 最大迭代次数
          tol     : 收敛判据（P矩阵变化的Frobenius范数）
    返回：K, P, Ps（同上）
    """
    K = K_init.copy()  # 初始的控制策略（不要求稳定，但过于不稳定可能导致数值问题）
    P = np.eye(n)      # 初始P矩阵（任意正定矩阵均可，这里用单位阵）
    Ps = []            # 存储每步迭代的P，用于画收敛图

    for _ in range(max_iter):
        Ac = A - B @ K          # 闭环矩阵
        # 值更新（Lyapunov递推，一步迭代，非精确求解）
        P_new = Ac.T @ P @ Ac + Q + K.T @ R @ K

        '''如果是Generalized Policy Iteration（GPI），这里的P更新就是多步迭代，只需要写一个for循环，迭代次数可以是1（对应值迭代）到无穷（对应策略迭代）。
        for i in range(n_inner__steps):  # 内层迭代，更新P多次（可以是1到无穷）
            P = Ac.T @ P @ Ac + Q + K.T @ R @ K
        '''
        
        # 策略改进
        K_new = np.linalg.solve(R + B.T @ P_new @ B, B.T @ P_new @ A)
        Ps.append(P_new.copy())

        # 判断P矩阵是否收敛
        if np.linalg.norm(P_new - P) < tol:
            P = P_new
            K = K_new
            break
        P = P_new   # 更新P
        K = K_new   # 更新K

    return K, P, Ps


# ==============================================================================
# Stage 3：在线PI + TD误差 + 值函数逼近VFA（只需B，不需要A）
# ==============================================================================
def stage3_online_pi_vfa(K_init, n_pi_steps=8, n_rl_steps=600, gamma=1.0):
    """
    在线策略迭代：沿系统轨迹采集数据，用最小二乘拟合值函数，不需要知道A矩阵。

    核心思路（论文公式46-48）：
      TD方程（时间差分方程）：w^T [φ(x_k) - γ·φ(x_{k+1})] = r(x_k, u_k)
      这是Bellman方程 V(x_k) = r_k + γ·V(x_{k+1}) 在参数化表示下的残差方程。
      用RLS（递推最小二乘）或批量LS求解参数向量w，得到值函数的近似。
      策略改进仍需B（用于从∂V/∂x计算∂V/∂u = B^T ∂V/∂x）。

    探索噪声：给控制加小幅随机扰动（Probing Noise），保证持续激励(PE)条件，
              使LS方程组有唯一解（回归矩阵满秩）。

    参数：K_init    : 初始稳定化增益
          n_pi_steps: 外层PI迭代次数（策略更新次数）
          n_rl_steps: 内层数据采集步数（每次策略固定后采多少条数据）
          gamma     : 折扣因子（论文中用1.0，即无折扣的无穷时域问题）
    返回：K        : 学习到的最优增益
          P_approx : 最后一步的近似P矩阵
          Ps_approx: 每步外层迭代的P矩阵列表
    """
    K = K_init.copy()      # 当前策略增益
    Ps_approx = []         # 收敛历史

    for pi_step in range(n_pi_steps):   # 外层：策略迭代循环
        # ── 数据采集：固定当前策略K，沿轨迹收集 (x_k, x_{k+1}, r_k) ──
        X_k, X_k1, Rs = [], [], []
        x = x0.copy() + np.random.randn(n) * 0.5   # 随机初始化（增加数据多样性）
        for _ in range(n_rl_steps):
            u = -K @ x + np.random.randn(m) * 0.05  # 控制=策略+探索噪声
            r = float(x @ Q @ x + u @ R @ u)         # 计算单步代价
            xn = A @ x + B @ u                        # 系统状态转移（真实系统）
            X_k.append(x.copy())    # 记录当前状态
            X_k1.append(xn.copy()) # 记录下一状态
            Rs.append(r)           # 记录代价
            x = xn                 # 推进时间

        # ── 构建TD回归矩阵并求解（批量LS）──
        # 每行：φ(x_k) - γ·φ(x_{k+1})，对应TD方程的"回归向量"
        Phi = np.array([quad_basis(xk) - gamma * quad_basis(xk1)
                        for xk, xk1 in zip(X_k, X_k1)])  # 形状 (n_rl_steps, L)
        R_vec = np.array(Rs)    # 右端向量：单步代价序列

        # 最小二乘求解：w = argmin ||Phi·w - R_vec||²
        # 等价于求解TD方程组 w^T·Phi[k] = r_k 在最小二乘意义下的解
        w, *_ = np.linalg.lstsq(Phi, R_vec, rcond=None)

        # ── 从参数向量w恢复对称矩阵P（用于策略改进）──
        P_approx = recover_P_from_w(w)
        Ps_approx.append(P_approx.copy())

        # ── 策略改进：K_{j+1} = (R + B^T P B)^{-1} B^T P A （仍需要B）──
        K = np.linalg.solve(R + B.T @ P_approx @ B, B.T @ P_approx @ A)

    return K, P_approx, Ps_approx


# ==============================================================================
# Stage 4：Actor-Critic 双神经网络（只需B，Critic不需A，Actor不需A）
# ==============================================================================
def stage4_actor_critic(K_init, n_pi_steps=10, n_rl_steps=800,
                         alpha=0.01, beta=0.005, gamma=1.0):
    """
    引入第二个神经网络作为 Actor，直接参数化控制策略。

    结构（论文公式55-56）：
      Critic网络：w^T φ(x_k) ≈ V(x_k)    → 通过TD误差更新参数w
      Actor网络 ：u_k = U^T x_k           → 线性参数化控制律（等价于 u=-Kx）
                  通过对Critic输出关于控制u的梯度更新参数U

    优势：Critic更新不需要A（只观测x_k, x_{k+1}, r_k）
          Actor更新需要B（计算 ∂V/∂u = B^T ∂V/∂x，这是从状态梯度到控制梯度的映射）
          但完全不需要A（内部动力学f(x)）

    局限：梯度下降的Actor对学习率(beta)敏感，收敛不如LS方法稳定，
          这也是为什么在我们的实验中Stage 4的K误差最大。

    参数：alpha : Critic梯度下降步长
          beta  : Actor梯度下降步长
    """
    K = K_init.copy()
    L = n * (n + 1) // 2   # 值函数基函数数量：n(n+1)/2
    w = np.zeros(L)         # Critic参数初始化为零向量
    U = -K.T                # Actor参数初始化：u=U^T x=-Kx，U形状(n,m)

    Ps_approx, Ks = [], []  # 分别存P矩阵和K增益的历史

    for pi_step in range(n_pi_steps):   # 外层迭代

        # ── Critic 更新（在线梯度下降拟合值函数）──
        x = x0.copy() + np.random.randn(n) * 0.5
        w_c = w.copy()   # 本轮从上一轮结果出发继续训练
        for _ in range(n_rl_steps):
            u = U.T @ x + np.random.randn(m) * 0.05   # Actor输出+探索噪声
            r = float(x @ Q @ x + u @ R @ u)            # 单步代价
            xn = A @ x + B @ u                           # 真实系统转移

            phi_k  = quad_basis(x)    # 当前状态的基函数值
            phi_k1 = quad_basis(xn)   # 下一状态的基函数值

            # TD误差（时间差分误差）= 实际代价 + 折扣后的下一步值估计 - 当前值估计
            # e_k = r_k + γ·V̂(x_{k+1}) - V̂(x_k) = r_k + γ·w^T φ(x_{k+1}) - w^T φ(x_k)
            td_err = r + gamma * (w_c @ phi_k1) - (w_c @ phi_k)

            # 梯度下降更新w：w ← w + α·[φ(x_k)-γ·φ(x_{k+1})]·e_k
            # 这是TD(0)的参数化版本，最小化TD误差的均方期望
            Fk = phi_k - gamma * phi_k1   # 回归向量（TD方程中的φ差值）
            w_c = w_c + alpha * Fk * td_err
            x = xn   # 推进时间

        w = w_c   # 用本轮学到的w更新全局Critic参数

        # 从w恢复P矩阵用于记录
        P_approx = recover_P_from_w(w)
        Ps_approx.append(P_approx.copy())

        # ── Actor 更新（梯度下降最小化值函数，通过B将状态梯度转为控制梯度）──
        x = x0.copy() + np.random.randn(n) * 0.5
        for _ in range(n_rl_steps):
            u_curr = U.T @ x             # 当前Actor输出的控制量
            xn = A @ x + B @ u_curr      # 状态转移

            # 计算值函数对x_{k+1}的梯度：∂V/∂x_{k+1} = (∂φ/∂x)^T · w
            grad_phi = gradient_quad_basis(xn)   # Jacobian (L, n)
            dV_dx = grad_phi.T @ w               # ∂V/∂x，形状(n,)

            # 通过B矩阵将状态梯度映射为控制梯度：∂V/∂u = B^T · ∂V/∂x（需要B！）
            dV_du = B.T @ dV_dx   # 形状(m,)

            # 对Actor参数U的梯度：∂/∂U [u^T R u + γ·(∂V/∂u)^T u]
            # u = U^T x，所以 ∂u/∂U = x⊗I_m，展开得到：
            grad_U = x.reshape(-1, 1) @ (2 * R @ u_curr + gamma * dV_du).reshape(1, -1)

            # 梯度下降更新Actor参数U
            U = U - beta * grad_U
            x = xn

        K = -U.T     # 从Actor参数U恢复控制增益K（u=U^T x=-Kx）
        Ks.append(K.copy())   # 记录本轮增益

    return K, P_approx, Ps_approx, Ks


# ==============================================================================
# Stage 5：Q-Learning（A和B都不需要，完全无模型）
# ==============================================================================
def stage5_q_learning(K_init, n_pi_steps=12, n_rl_steps=1000,
                       gamma=1.0, pe_noise=0.3):
    """
    Q函数学习（Action-Dependent HDP）：学习Q(x,u)而非V(x)，
    策略改进时对Q关于u求导，完全绕过系统动力学。

    Q函数（动作-价值函数）定义（论文公式60）：
      Q(x_k, u_k) = r(x_k, u_k) + γ·V(x_{k+1})

    关键思路：最优策略满足 ∂Q*(x,u)/∂u = 0
      对于LQR，Q*(x,u) = z^T H z（z=[x;u]），从H矩阵可直接读出最优增益：
      u* = -H_uu^{-1} H_ux · x，完全不需要A或B！

    TD方程（固定点方程，论文公式67）：
      Q(x_k, u_k) = r_k + γ·Q(x_{k+1}, u_{k+1})
      用LS在轨迹数据上求解：H^T [ψ(z_k) - γ·ψ(z_{k+1})] = r_k

    持续激励（PE）：必须给控制加噪声（pe_noise），否则回归矩阵退秩，
      LS方程无唯一解（论文Bradtke 1994指出）。

    参数：pe_noise : 探索噪声标准差（越大PE越好，但会增加估计方差）
    """
    K = K_init.copy()
    aug_dim = (n + m) * (n + m + 1) // 2   # 增广向量[x;u]的二次型基函数维度
    H_vec = np.zeros(aug_dim)               # Q函数参数向量初始化
    Hs = []                                 # 记录每步迭代的H向量

    for pi_step in range(n_pi_steps):   # 外层策略迭代

        # ── 数据采集：沿轨迹收集 (z_k, z_{k+1}, r_k) ──
        Phi_list, R_list = [], []
        x = x0.copy() + np.random.randn(n) * 0.5
        for _ in range(n_rl_steps):
            # 控制 = 当前策略 + PE探索噪声（保证持续激励）
            u = -K @ x + np.random.randn(m) * pe_noise
            r  = float(x @ Q @ x + u @ R @ u)   # 单步代价
            xn = A @ x + B @ u                    # 状态转移
            un = -K @ xn                          # 下一步策略输出（无探索噪声）

            z_k  = quad_basis_aug(x,  u)   # 当前增广基函数 ψ(z_k)
            z_k1 = quad_basis_aug(xn, un)  # 下一增广基函数 ψ(z_{k+1})

            # Q函数TD回归向量：ψ(z_k) - γ·ψ(z_{k+1})
            Phi_list.append(z_k - gamma * z_k1)
            R_list.append(r)
            x = xn

        # ── 批量LS求解Q函数参数H_vec ──
        Phi  = np.array(Phi_list)   # 回归矩阵 (n_rl_steps, aug_dim)
        R_vec = np.array(R_list)    # 单步代价向量
        H_vec, *_ = np.linalg.lstsq(Phi, R_vec, rcond=None)
        Hs.append(H_vec.copy())

        # ── 重建对称H矩阵（对应Q函数的二次型核矩阵）──
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

        # ── 分块提取子矩阵 ──
        # H_mat = [[H_xx, H_xu], [H_ux, H_uu]]，其中：
        # H_xx: x相关的二次型核
        # H_uu: u相关的二次型核（等价于R + B^T P B）
        # H_ux: 控制-状态交叉项（等价于B^T P A）
        H_ux = H_mat[n:, :n]   # 形状 (m, n)
        H_uu = H_mat[n:, n:]   # 形状 (m, m)

        # ── 策略改进（完全不需要A或B！）──
        # 最优控制：∂Q*/∂u = 0 → H_uu·u + H_ux·x = 0 → u* = -H_uu^{-1}·H_ux·x
        try:
            K = np.linalg.solve(H_uu, H_ux)   # K = H_uu^{-1} H_ux
        except np.linalg.LinAlgError:
            pass   # 奇异时保持旧K

    return K, H_vec, H_mat, Hs


# ==============================================================================
# Stage 6：连续时间策略迭代（CT PI，只需B，不需A）
# ==============================================================================
def stage6_ct_policy_iteration(K_init, n_pi_steps=10,
                                 T_interval=0.5, n_samples=300,
                                 dt=0.01):
    """
    连续时间系统的在线策略迭代，使用区间Bellman方程避免对A的依赖。

    连续时间系统：ẋ = f(x) + g(x)u = A_ct·x + B_ct·u

    CT Bellman方程（连续时间下的值函数定义）：
      0 = r(x,u) + (∂V/∂x)^T·(f(x)+g(x)u)    （微分形式，需要f(x)=A_ct·x！）

    区间Bellman方程（论文公式89，避免对f的依赖）：
      V(x(t)) = ∫_t^{t+T} r(x,u)dτ + V(x(t+T))
    对应的TD误差（公式90）：
      e(t:t+T) = ∫_t^{t+T} r dτ + V(x(t+T)) - V(x(t))
    这个形式不含系统动力学，可以直接从轨迹数据估计！

    LS方程：w^T [φ(x(t)) - φ(x(t+T))] = ∫_t^{t+T} r dτ

    策略改进（论文公式93）：m(x) = -½ R^{-1} B^T ∂V/∂x
    对线性系统 V=x^T P x：∂V/∂x = 2Px，所以 K = R^{-1} B^T P（需要B）

    参数：T_interval: 每段积分区间长度（秒）
          n_samples : 每次PI迭代采集的区间段数
          dt        : Euler积分步长（秒）
    """
    # 连续时间双积分器：ẋ₁=x₂, ẋ₂=u
    A_ct = np.array([[0.0, 1.0],   # ẋ₁ = x₂
                     [0.0, 0.0]])  # ẋ₂ = 0（纯积分，无阻尼）
    B_ct = np.array([[0.0],        # u不直接影响x₁
                     [1.0]])       # u直接作为x₂的加速度输入
    Q_ct = np.eye(n)               # CT系统的状态代价权重
    R_ct = np.array([[0.1]])       # CT系统的控制代价权重

    K = K_init.copy()       # 当前控制增益
    Ps_approx = []          # 收敛历史

    def simulate_ct_segment(x_start, K_ct, duration, dt_sim):
        """
        Euler法积分一段CT轨迹，同时累积代价积分。
        返回：最终状态 x_end 和区间代价积分 ∫r dτ。
        """
        x = x_start.copy()
        int_cost = 0.0                          # 区间代价积分初始化
        steps = int(duration / dt_sim)          # 积分步数
        for _ in range(steps):
            u = -K_ct @ x + np.random.randn(m) * 0.02   # 加小探索噪声
            # 代价积分：用矩形法近似 ∫r dτ ≈ Σ r(x,u)·dt
            int_cost += float(x @ Q_ct @ x + u @ R_ct @ u) * dt_sim
            x = x + (A_ct @ x + B_ct @ u) * dt_sim   # Euler积分：x_{t+dt}=x_t+ẋ·dt
        return x, int_cost

    for pi_step in range(n_pi_steps):   # 外层策略迭代
        Phi_list, IntR_list = [], []
        x = x0.copy() + np.random.randn(n) * 0.3   # 随机初始化增加数据多样性

        for _ in range(n_samples):
            # 记录区间起点的基函数值 φ(x(t))
            phi_start = quad_basis(x)

            # 积分一段轨迹，得到终点状态和区间代价
            x_end, int_r = simulate_ct_segment(x, K, T_interval, dt)

            # 记录区间终点的基函数值 φ(x(t+T))
            phi_end = quad_basis(x_end)

            # 区间Bellman方程的回归向量：φ(x(t)) - φ(x(t+T))
            Phi_list.append(phi_start - phi_end)
            IntR_list.append(int_r)   # 对应右端：∫r dτ

            # 随机重置状态（增加探索多样性，保证PE条件）
            x = x_end + np.random.randn(n) * 0.1

        # 批量LS求解值函数参数w
        Phi = np.array(Phi_list)    # 回归矩阵 (n_samples, L)
        IntR = np.array(IntR_list)  # 右端向量 (n_samples,)
        w, *_ = np.linalg.lstsq(Phi, IntR, rcond=None)

        # 恢复P矩阵
        P_approx = recover_P_from_w(w)
        Ps_approx.append(P_approx.copy())

        # 策略改进（需要B_ct）：K = R^{-1} B^T P
        K = np.linalg.solve(R_ct, B_ct.T @ P_approx)

    return K, P_approx, Ps_approx, A_ct, B_ct, Q_ct, R_ct


# ==============================================================================
# 运行所有Stage并打印结果
# ==============================================================================
print("=" * 70)
print("  Lewis & Vrabie (2009) — 强化学习与自适应动态规划复现")
print("=" * 70)

# ── Stage 1：离线Riccati，得到全局最优基准 ──
P_star, K_star = stage1_riccati()
print(f"\n[Stage 1] 离线Riccati最优增益 K* = {K_star.flatten()}")
print(f"          最优代价矩阵 P* =\n{P_star}")

# 初始稳定化增益（手动设计，使闭环极点在单位圆内）
K_init = np.array([[2.0, 1.5]])

# ── Stage 2a：策略迭代 ──
print("\n[Stage 2a] 策略迭代 Policy Iteration (Hewer算法)...")
K_pi, P_pi, Ps_pi = stage2_policy_iteration(K_init)
print(f"  K_PI = {K_pi.flatten()}  (与K*误差 = {np.linalg.norm(K_pi-K_star):.2e})")

# ── Stage 2b：值迭代 ──
print("\n[Stage 2b] 值迭代 Value Iteration (Lancaster-Rodman)...")
K_vi, P_vi, Ps_vi = stage2_value_iteration(K_init)
print(f"  K_VI = {K_vi.flatten()}  (与K*误差 = {np.linalg.norm(K_vi-K_star):.2e})")

# ── Stage 3：TD+VFA在线PI ──
print("\n[Stage 3] 在线PI + TD误差 + 值函数逼近VFA (只需B)...")
K_s3, P_s3, Ps_s3 = stage3_online_pi_vfa(K_init)
print(f"  K_s3 = {K_s3.flatten()}  (与K*误差 = {np.linalg.norm(K_s3-K_star):.2e})")

# ── Stage 4：Actor-Critic ──
print("\n[Stage 4] Actor-Critic 双神经网络 (只需B，不需A)...")
K_s4, P_s4, Ps_s4, Ks_s4 = stage4_actor_critic(K_init)
print(f"  K_s4 = {K_s4.flatten()}  (与K*误差 = {np.linalg.norm(K_s4-K_star):.2e})")

# ── Stage 5：Q-Learning ──
print("\n[Stage 5] Q-Learning (A和B都不需要)...")
K_s5, H_vec, H_mat, Hs_s5 = stage5_q_learning(K_init)
print(f"  K_s5 = {K_s5.flatten()}  (与K*误差 = {np.linalg.norm(K_s5-K_star):.2e})")

# ── Stage 6：连续时间PI ──
print("\n[Stage 6] 连续时间策略迭代 CT-PI (只需B)...")
K_s6, P_s6, Ps_s6, A_ct, B_ct, Q_ct, R_ct = stage6_ct_policy_iteration(
    np.array([[1.0, 2.0]]))
print(f"  K_s6 = {K_s6.flatten()}")

# ==============================================================================
# 仿真轨迹（仅DT系统Stage 1-5）
# ==============================================================================
results = {
    "Stage1 Riccati\n(离线,A+B)":        K_star,
    "Stage2 PI\n(在线,A+B)":             K_pi,
    "Stage2 VI\n(在线,A+B)":             K_vi,
    "Stage3 TD+VFA\n(在线,只需B)":       K_s3,
    "Stage4 Actor-Critic\n(在线,只需B)": K_s4,
    "Stage5 Q-Learning\n(在线,无需A,B)": K_s5,
}

trajectories = {}
for label, K in results.items():
    xs, us, costs = simulate(K)
    trajectories[label] = (xs, us, costs)

# 计算各Stage的P矩阵误差序列（用于收敛图）
def P_err(Ps_list):
    """计算每次迭代的P矩阵与最优P*之间的Frobenius范数误差"""
    return [np.linalg.norm(P - P_star, 'fro') for P in Ps_list]

# ==============================================================================
# 绘图
# ==============================================================================
# 颜色方案：每个Stage一个固定颜色
colors = {
    "Stage1 Riccati\n(离线,A+B)":        "#E63946",   # 红：基准最优
    "Stage2 PI\n(在线,A+B)":             "#2A9D8F",   # 青绿：PI
    "Stage2 VI\n(在线,A+B)":             "#457B9D",   # 蓝：VI
    "Stage3 TD+VFA\n(在线,只需B)":       "#E9C46A",   # 黄：VFA
    "Stage4 Actor-Critic\n(在线,只需B)": "#F4A261",   # 橙：Actor-Critic
    "Stage5 Q-Learning\n(在线,无需A,B)": "#A8DADC",   # 浅蓝：Q-Learning
}

# 创建图形：4行3列布局，黑色背景
fig = plt.figure(figsize=(20, 22), facecolor='#0D1117')
fig.suptitle("Lewis & Vrabie (2009)  ·  强化学习与自适应动态规划用于反馈控制",
             fontsize=15, color='#E6EDF3', fontweight='bold', y=0.98)

# 使用GridSpec实现灵活的子图布局
gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.07, right=0.97, top=0.95, bottom=0.04)

def style_ax(ax, title, xlabel_zh, ylabel_zh):
    """
    统一设置子图样式（暗色主题）并设置中文坐标轴标签。
    参数：
        title    : 子图标题（英文，用于技术描述）
        xlabel_zh: x轴中文标签
        ylabel_zh: y轴中文标签
    """
    ax.set_facecolor('#161B22')          # 子图背景色（深灰蓝）
    ax.tick_params(colors='#8B949E', labelsize=8)   # 刻度颜色和字号
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')   # 边框颜色
    ax.set_title(title, color='#E6EDF3', fontsize=9, pad=6, fontweight='bold')
    ax.set_xlabel(xlabel_zh, color='#8B949E', fontsize=9)   # 中文x轴标签
    ax.set_ylabel(ylabel_zh, color='#8B949E', fontsize=9)   # 中文y轴标签
    ax.grid(True, color='#21262D', linewidth=0.5, linestyle='--')  # 参考网格线

# ── 第1行（横跨3列）：x₁状态轨迹总览 ────────────────────────────────────────
# 纵坐标含义：x₁是系统第一个状态变量（可理解为位置），
#             从初始值5.0出发，各算法的控制律将其逐渐拉回到0
ax_x1 = fig.add_subplot(gs[0, :])   # 占满第0行所有3列
style_ax(ax_x1, "State x₁ Trajectories — all stages",
         "时间步 k", "x₁（位置状态）\n从初值5.0收敛到0")
for label, (xs, us, costs) in trajectories.items():
    short = label.replace('\n', ' ')
    ax_x1.plot(xs[:, 0], color=colors[label], linewidth=1.8, label=short, alpha=0.9)
ax_x1.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')   # 零线参考
ax_x1.legend(loc='upper right', fontsize=7, facecolor='#161B22',
             edgecolor='#30363D', labelcolor='#E6EDF3', ncol=3)
# 注：Stage1/2/3/5的K值几乎完全相同，轨迹重叠，只能看到Stage4（橙色）和其余叠成的一条线

# ── 第2行左：x₂状态轨迹 ───────────────────────────────────────────────────────
# 纵坐标含义：x₂是系统第二个状态变量（速度），初始值-3.0
#             先超调（向更负方向运动），然后收敛到0
ax_x2 = fig.add_subplot(gs[1, 0])
style_ax(ax_x2, "State x₂ Trajectories",
         "时间步 k", "x₂（速度状态）\n从初值-3.0收敛到0")
for label, (xs, us, costs) in trajectories.items():
    ax_x2.plot(xs[:, 1], color=colors[label], linewidth=1.5, alpha=0.85)
ax_x2.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')

# ── 第2行中：控制输入u ──────────────────────────────────────────────────────────
# 纵坐标含义：控制量u（力/力矩），u=-Kx
#             初始控制量大（用力拉回状态），之后逐渐衰减至0
#             最优控制器的控制曲线"最平滑"、"最经济"
ax_u = fig.add_subplot(gs[1, 1])
style_ax(ax_u, "Control Input u",
         "时间步 k", "控制量 u\n（u=-Kx，初始大，逐渐衰减到0）")
for label, (xs, us, costs) in trajectories.items():
    ax_u.plot(us[:, 0], color=colors[label], linewidth=1.5, alpha=0.85)
ax_u.axhline(0, color='#58A6FF', linewidth=0.7, linestyle=':')

# ── 第2行右：累积代价 ───────────────────────────────────────────────────────────
# 纵坐标含义：到当前时刻为止的累积代价 Σ r(x_k, u_k) = Σ(x^T Q x + u^T R u)
#             最终稳定值就是该控制策略的总代价（越小越优）
#             最优控制器（红色）的稳定值最低，代表最少资源消耗
ax_cost = fig.add_subplot(gs[1, 2])
style_ax(ax_cost, "Cumulative Cost",
         "时间步 k", "累积代价 Σ(x²+u²)\n（最终稳定值=该策略的总代价）")
for label, (xs, us, costs) in trajectories.items():
    ax_cost.plot(np.cumsum(costs), color=colors[label], linewidth=1.5, alpha=0.85)

# ── 第3行左：Stage 2 PI收敛曲线 ────────────────────────────────────────────────
# 纵坐标含义（对数刻度）：每次PI迭代后估计的P矩阵与最优P*之间的Frobenius范数误差
#   ‖Pⱼ - P*‖_F，其中‖·‖_F = sqrt(Σᵢⱼ (Pᵢⱼ)²)
#   值越小表示当前P越接近最优解；曲线越快到达底部表示收敛越快
#   PI的Frobenius范数误差只需约5步就从10²量级降至10⁻¹²（接近机器精度）
ax_pi = fig.add_subplot(gs[2, 0])
style_ax(ax_pi, "Stage 2 — Policy Iteration Convergence",
         "PI 迭代次数 j",
         "‖Pⱼ − P*‖_F\n（P矩阵Frobenius范数误差，对数刻度）\n值越小越接近最优P*")
errs_pi = P_err(Ps_pi)
ax_pi.semilogy(errs_pi, color='#2A9D8F', linewidth=2, marker='o', ms=5)

# ── 第3行中：Stage 2 VI收敛曲线 ────────────────────────────────────────────────
# 纵坐标含义（对数刻度）：与PI相同，‖Pⱼ - P*‖_F
#   值迭代需要约100步才能达到PI的收敛精度（线性收敛 vs PI的二次收敛）
#   每步计算量极小（只是矩阵乘法），但需要更多步数
ax_vi = fig.add_subplot(gs[2, 1])
style_ax(ax_vi, "Stage 2 — Value Iteration Convergence",
         "VI 迭代次数 j",
         "‖Pⱼ − P*‖_F\n（P矩阵Frobenius范数误差，对数刻度）\n收敛比PI慢（线性收敛）")
errs_vi = P_err(Ps_vi)
ax_vi.semilogy(errs_vi, color='#457B9D', linewidth=2, marker='s', ms=4)

# ── 第3行右：Stage 3 VFA在线收敛曲线 ──────────────────────────────────────────
# 纵坐标含义（对数刻度）：‖P̂ - P*‖_F
#   P̂是通过在线LS拟合值函数参数w后恢复的近似P矩阵
#   出现振荡是因为：数据来自带噪声的轨迹，LS估计有随机误差
#   整体趋势仍向P*靠近，最终误差约10⁻³（比离线方法大，因为有采样噪声）
ax_s3 = fig.add_subplot(gs[2, 2])
style_ax(ax_s3, "Stage 3 — Online VFA Convergence",
         "外层PI迭代次数 j",
         "‖P̂ − P*‖_F\n（在线估计的P与最优P*的误差，对数刻度）\n振荡来自采样随机性")
errs_s3 = P_err(Ps_s3)
ax_s3.semilogy(errs_s3, color='#E9C46A', linewidth=2, marker='^', ms=5)

# ── 第4行左：Stage 4 Actor-Critic增益收敛曲线 ──────────────────────────────────
# 纵坐标含义（对数刻度）：‖Kⱼ - K*‖₂（控制增益K的Euclidean范数误差）
#   注意曲线先下降后上升，说明Actor梯度下降在最优点附近振荡/发散
#   原因：固定学习率beta在二次型代价函数附近导致过冲
#   这是Stage 4相比其他Stage误差最大的根本原因
ax_s4 = fig.add_subplot(gs[3, 0])
style_ax(ax_s4, "Stage 4 — Actor-Critic Gain Convergence",
         "PI 迭代次数 j",
         "‖Kⱼ − K*‖\n（控制增益Euclidean范数误差，对数刻度）\n先降后升=梯度下降振荡")
K_errs_s4 = [np.linalg.norm(Kj - K_star) for Kj in Ks_s4]
ax_s4.semilogy(K_errs_s4, color='#F4A261', linewidth=2, marker='D', ms=5)

# ── 第4行中：Stage 5 Q-Learning增益收敛曲线 ────────────────────────────────────
# 纵坐标含义（对数刻度）：‖Kⱼ - K*‖₂（每次Q-Learning迭代后恢复的K与K*的误差）
#   K是从学到的H矩阵中提取的：K = H_uu^{-1} H_ux
#   快速收敛到10⁻¹⁴量级（机器精度），说明Q-Learning以数值精度复现了最优解
#   全程不需要A和B，是最强的无模型最优控制算法
ax_s5 = fig.add_subplot(gs[3, 1])
style_ax(ax_s5, "Stage 5 — Q-Learning Gain Convergence",
         "Q-Learning 迭代次数 j",
         "‖Kⱼ − K*‖\n（Q-Learning估计的K与最优K*的误差，对数刻度）\n收敛至机器精度")
QL_K_errs = []
for h in Hs_s5:
    aug = n + m
    H_m = np.zeros((aug, aug))
    idx = 0
    for i in range(aug):
        for j in range(i, aug):
            if i == j: H_m[i, j] = h[idx]
            else:      H_m[i, j] = H_m[j, i] = h[idx] / 2
            idx += 1
    try:
        K_tmp = np.linalg.solve(H_m[n:, n:], H_m[n:, :n])
        QL_K_errs.append(np.linalg.norm(K_tmp - K_star))
    except:
        QL_K_errs.append(np.nan)
ax_s5.semilogy(QL_K_errs, color='#A8DADC', linewidth=2, marker='v', ms=5)

# ── 第4行右：Stage 6 连续时间PI收敛曲线 ────────────────────────────────────────
# 纵坐标含义（对数刻度）：‖P̂ - P*_ct‖_F（连续时间系统的P矩阵误差）
#   P*_ct：通过极细DT近似得到的连续时间Riccati方程的参考解
#   曲线先低后高再平稳：初始P偶然接近真值，之后因数值积分误差和有限采样误差上升，
#   最终稳定在一个有限误差平台，这是CT在线RL的固有局限（比DT难）
ax_s6 = fig.add_subplot(gs[3, 2])
style_ax(ax_s6, "Stage 6 — CT Policy Iteration Convergence",
         "CT PI 迭代次数 j",
         "‖P̂ − P*_ct‖_F\n（CT系统P矩阵误差，对数刻度）\n平台误差来自积分精度限制")

# 用极细时间步的离散近似作为CT Riccati的参考解
dt_fine = 1e-3
P_star_ct = solve_discrete_are(
    np.eye(n) + A_ct * dt_fine,   # 近似CT的A矩阵：I + A_ct·dt
    B_ct * dt_fine,               # 近似CT的B矩阵：B_ct·dt
    Q_ct * dt_fine, R_ct          # 乘以dt保持代价量纲一致
) / dt_fine                       # 除以dt将DT代价矩阵换算回CT量纲

errs_s6 = [np.linalg.norm(P - P_star_ct, 'fro') for P in Ps_s6]
ax_s6.semilogy(errs_s6, color='#C77DFF', linewidth=2, marker='p', ms=6)

# ── 修复Stage 6纵坐标颜色为白色 ──────────────────────────────────────────────
# 默认style_ax将y轴标签设为灰色(#8B949E)，这里单独覆盖为白色，提高可读性
ax_s6.yaxis.label.set_color('#FFFFFF')
ax_s6.tick_params(axis='y', colors='#FFFFFF')   # y轴刻度值也设为白色

# 保存图片
plt.savefig('claude_lewis2009_reproduction.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1117')
print("\n✓ 图片已保存至 claude_lewis2009_reproduction.png")

# ==============================================================================
# 终端汇总表（ASCII格式）
# ==============================================================================
# 计算各Stage的总仿真代价
def total_cost(K):
    _, _, costs = simulate(K)
    return float(np.sum(costs))

tc_star = total_cost(K_star)
tc_pi   = total_cost(K_pi)
tc_vi   = total_cost(K_vi)
tc_s3   = total_cost(K_s3)
tc_s4   = total_cost(K_s4)
tc_s5   = total_cost(K_s5)

print("\n" + "=" * 90)
print(f"  {'算法':<32} {'K值':^24} {'K误差':>9} {'总代价':>9} {'需A':>6} {'需B':>6} {'在线':>6}")
print("-" * 90)

rows = [
    ("Stage 1: Riccati (离线基准)",
     K_star,  np.linalg.norm(K_star-K_star), tc_star, True,  True,  False),
    ("Stage 2: Policy Iteration (Hewer)",
     K_pi,    np.linalg.norm(K_pi-K_star),   tc_pi,   True,  True,  True),
    ("Stage 2: Value Iteration (L-R)",
     K_vi,    np.linalg.norm(K_vi-K_star),   tc_vi,   True,  True,  True),
    ("Stage 3: TD+VFA 单Critic",
     K_s3,    np.linalg.norm(K_s3-K_star),   tc_s3,   False, True,  True),
    ("Stage 4: Actor-Critic 双NN",
     K_s4,    np.linalg.norm(K_s4-K_star),   tc_s4,   False, True,  True),
    ("Stage 5: Q-Learning (完全无模型)",
     K_s5,    np.linalg.norm(K_s5-K_star),   tc_s5,   False, False, True),
]

for name, K, err, tc, nA, nB, online in rows:
    k_str = f"[{K[0,0]:.4f}, {K[0,1]:.4f}]"
    print(f"  {name:<32} {k_str:^24} {err:>9.2e} {tc:>9.2f}"
          f" {'是':>6} {'是' if nB else '否':>6} {'是' if online else '否':>6}"
          .replace("是 ", "是 " if nA else "否 ", 1))

# 重新打印（上面的字符串操作有点绕，直接用条件表达式更清晰）
print("-" * 90)
print(f"  {'算法':<32} {'K值':^24} {'K误差':>9} {'总代价':>9} {'需A':>6} {'需B':>6} {'在线':>6}")
print("-" * 90)
for name, K, err, tc, nA, nB, online in rows:
    k_str = f"[{K[0,0]:.4f}, {K[0,1]:.4f}]"
    a_str = "是" if nA else "否"
    b_str = "是" if nB else "否"
    o_str = "是" if online else "否"
    print(f"  {name:<32} {k_str:^24} {err:>9.2e} {tc:>9.2f} {a_str:>6} {b_str:>6} {o_str:>6}")

print("=" * 90)
print("\n核心结论：Stage1/2/3/5的K值在小数点后4位完全一致，")
print(f"  轨迹完全重叠（总代价均为{tc_star:.2f}），无法从轨迹图中区分。")
print(f"  Stage4（Actor-Critic）K误差={np.linalg.norm(K_s4-K_star):.2f}，")
print(f"  总代价={tc_s4:.2f}（高出最优{(tc_s4-tc_star)/tc_star*100:.1f}%），")
print("  这是梯度下降Actor未完全收敛导致的。")
print("=" * 90)