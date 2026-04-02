import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import matplotlib.pyplot as plt
import copy

# ========================================================
# 模块零：全局设置与上帝视角环境
# ========================================================

def set_seed(seed=42):
    """
    固定所有随机种子，确保每次运行的轨迹完全一致。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class VrabieLewisEnv:
    """
    [来源: 物理环境 (Neural Networks 2009), Example 2]
    这是为了测试而专门构造的非线性系统。
    它不仅包含极度复杂的三角函数交叉项，更重要的是，作者给出了它的绝对最优解，
    方便我们验证神经网络是否真的收敛到了“真理”。
    """
    def __init__(self, dt=0.05):
        self.state_dim = 2       # 状态 x 包含两个维度: x1, x2
        self.action_dim = 1      # 动作 u 是标量 (单输入系统)
        self.dt = dt             # 连续系统的离散化步长，用于欧拉数值积分
        self.x = np.zeros((self.state_dim, 1)) 
        
        # Q矩阵: 对状态偏离的惩罚权重 (单位阵表示 x1 和 x2 同等重要)
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # R矩阵: 对控制能量消耗的惩罚权重
        self.R = np.array([[1.0]], dtype=np.float32)
        
        # A_lin 和 B_lin 是非线性系统在原点 (0,0) 处的局部线性化雅可比矩阵。
        # 它们仅供 模块1(LQR) 和 模块2(RLS) 使用，代表了经典控制理论的极限。
        self.A_lin = np.array([[-1.0, 1.0], [-0.5, 4.0]], dtype=np.float32)
        self.B_lin = np.array([[0.0], [3.0]], dtype=np.float32)

    def reset(self, x0=None):
        # 初始化状态。均匀采样范围设为 [-1.5, 1.5]
        if x0 is None:
            self.x = np.random.uniform([[-1.5], [-1.5]], [[1.5], [1.5]])
        else:
            self.x = np.array(x0, dtype=np.float32).reshape(self.state_dim, 1)
        return self.x.copy()

    def calc_cost(self, x, u):
        # 计算瞬时物理消耗: r(x, u) = x^T * Q * x + u^T * R * u
        return (x.T @ self.Q @ x + u.T @ self.R @ u).item()

    def optimal_control(self, x):
        # 理论上的绝对最优控制律 (上帝答案)。
        # 作者通过逆向求解 HJB 方程得出，用于最终画图对比。
        x1, x2 = x[0, 0], x[1, 0]
        return np.array([[- (np.cos(2*x1) + 2) * x2]], dtype=np.float32)

    def step(self, u):
        u = np.array(u).reshape(self.action_dim, 1)
        cost = self.calc_cost(self.x, u)
        x1, x2 = self.x[0, 0], self.x[1, 0]
        
        # 非线性动力学: dot_x = f(x) + g(x)u
        # 这里包含了导致传统线性控制器失效的 cos(2*x1) 项
        dot_x1 = -x1 + x2
        dot_x2 = -0.5*x1 - 0.5*x2*(1 - (np.cos(2*x1)+2)**2) + (np.cos(2*x1)+2)*u[0,0]
        
        # 一阶欧拉积分向前推演物理世界
        self.x[0, 0] += dot_x1 * self.dt
        self.x[1, 0] += dot_x2 * self.dt
        
        # 注意：返回的 cost 乘以了 dt，将其化为了这段时间内的代价积分矩形面积
        return self.x.copy(), cost * self.dt

def torch_vrabie_dynamics(x, u):
    # 这是供 PyTorch 自动求导机制使用的一份物理法则拷贝。
    # 在 M3 和 M5 中，我们需要让梯度“穿透”环境动力学反传给 Actor。
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    dot_x1 = -x1 + x2
    dot_x2 = -0.5*x1 - 0.5*x2*(1 - (torch.cos(2*x1)+2)**2) + (torch.cos(2*x1)+2)*u
    return torch.cat([dot_x1, dot_x2], dim=1)


# ========================================================
# 验证与可视化工具 (略过注释，核心逻辑是分别跑 Opt, LQR 和 Actor 画图)
# ========================================================
def evaluate_and_plot(env: VrabieLewisEnv, K_star, actor_net, module_name=""):
    x0 = np.array([[1.5], [-1.0]]) 
    steps = 100
    
    # 1. 运行解析最优
    env.reset(x0)
    opt_traj = [env.x.copy()]
    opt_cost = 0.0
    for _ in range(steps):
        u = env.optimal_control(env.x)
        x_next, cost = env.step(u)
        opt_traj.append(x_next)
        opt_cost += cost
        
    # 2. 运行 LQR
    env.reset(x0)
    lqr_traj = [env.x.copy()]
    lqr_cost = 0.0
    for _ in range(steps):
        u = -K_star @ env.x
        x_next, cost = env.step(u)
        lqr_traj.append(x_next)
        lqr_cost += cost
        
    # 3. 运行神经网络 Actor
    env.reset(x0)
    actor_traj = [env.x.copy()]
    actor_cost = 0.0
    for _ in range(steps):
        if actor_net is None: break
        x_tensor = torch.FloatTensor(env.x.T)
        with torch.no_grad():
            u = actor_net(x_tensor).numpy().T
        x_next, cost = env.step(u)
        actor_traj.append(x_next)
        actor_cost += cost
        
    opt_traj = np.array(opt_traj).squeeze()
    lqr_traj = np.array(lqr_traj).squeeze()
    actor_traj = np.array(actor_traj).squeeze()
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"[{module_name}] Opt Cost:{opt_cost:.2f} | LQR Cost:{lqr_cost:.2f} | Actor Cost:{actor_cost:.2f}")
    
    plt.subplot(1, 2, 1)
    plt.plot(opt_traj[:, 0], label='Opt. Nonlinear x1', color='black', linewidth=3, alpha=0.5)
    plt.plot(lqr_traj[:, 0], label='LQR x1', linestyle='--')
    if actor_net is not None:
        plt.plot(actor_traj[:, 0], label='Actor x1', alpha=0.9, linestyle='-.')
    plt.title("State x1 Trajectory")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(opt_traj[:, 0], opt_traj[:, 1], label='Opt. Phase', color='black', linewidth=3, alpha=0.5)
    plt.plot(lqr_traj[:, 0], lqr_traj[:, 1], label='LQR Phase', linestyle='--', marker='o', markersize=2)
    if actor_net is not None:
        plt.plot(actor_traj[:, 0], actor_traj[:, 1], label='Actor Phase', marker='x', markersize=3, linestyle='-.')
    plt.title("Phase Portrait (x1 vs x2)")
    plt.legend()
    plt.grid()
    
    filename = f"{module_name.replace(' ', '_')}_validation.png"
    plt.savefig(filename)
    plt.close()
    print(f"[{module_name}] 验证图生成: 解析最优={opt_cost:.2f} | LQR={lqr_cost:.2f} | 算法Actor={actor_cost:.2f}")


# ========================================================
# 模块一：离线 LQR (古典控制的终极真理)
# ========================================================
def module1_offline_lqr(env: VrabieLewisEnv):
    # 调用 scipy 底层求解 Continuous Algebraic Riccati Equation (CARE)
    # 提供一个基准矩阵 P_star 和最优线性增益 K_star
    P = scipy.linalg.solve_continuous_are(env.A_lin, env.B_lin, env.Q, env.R)
    K = np.linalg.inv(env.R) @ env.B_lin.T @ P
    print("\n--- 模块一：离线 LQR ---")
    print(f"由雅可比矩阵解得的 P 矩阵:\n{P}")
    print(f"局部线性化的 K 矩阵:\n{K}")
    return P, K


# ========================================================
# 模块二：在线 Policy Iteration (基于闭式最小二乘)
# ========================================================
def phi_rls(x):
    # 因为要证明数据驱动等价于解黎卡提方程，所以严格使用纯二次型基底
    # 包含了对称矩阵的所有独立元素 [x1^2, 2*x1*x2, x2^2]
    x1, x2 = x[0,0], x[1,0]
    return np.array([[x1**2], [2*x1*x2], [x2**2]])

def get_P_from_W(W):
    # 将解出的权重向量 W 拉伸还原为 2x2 的对称矩阵 P
    w1, w2, w3 = W.flatten()
    return np.array([[w1, w2], [w2, w3]])

def module2_online_rls_policy_iteration(env: VrabieLewisEnv, K_init, K_star):
    print("\n--- 模块二：在线 Policy Iteration (基于闭式最小二乘，证明 LQR 等价性) ---")
    K_i = np.array(K_init, dtype=np.float32)
    
    # 策略迭代外循环
    for i in range(1, 11):
        X_mat = []
        Y_mat = []
        # 空间随机采样 500 个点，收集方程组所需的数据
        for _ in range(500):
            env.x = np.random.uniform([[-1.5], [-1.5]], [[1.5], [1.5]])
            x = env.x.copy()
            x1_val, x2_val = x[0,0], x[1,0]
            u_i = -K_i @ x
            
            # 【核心逻辑】：为了在数学上严格等价于 LQR，这里必须强制使用局部线性化动力学。
            # 否则纯二次型基底无法拟合非线性环境中的三角函数。
            dot_x = env.A_lin @ x + env.B_lin @ u_i
            dot_x1, dot_x2 = dot_x[0,0], dot_x[1,0]
            
            # 根据链式法则算出特征基底随时间的导数 d_phi / dt
            dot_phi = np.array([
                [2 * x1_val * dot_x1],
                [x1_val * dot_x2 + dot_x1 * x2_val],
                [2 * x2_val * dot_x2]
            ])
            
            cost_r = env.calc_cost(x, u_i)
            # 组装超定方程组 W^T * dot_phi = -r
            X_mat.append(dot_phi.T)
            Y_mat.append(-cost_r) 
            
        X_mat = np.vstack(X_mat)
        Y_mat = np.array(Y_mat).reshape(-1, 1)
        
        # 使用伪逆求解最小二乘 (策略评估)
        W_i_plus_1 = np.linalg.pinv(X_mat) @ Y_mat
        P_hat = get_P_from_W(W_i_plus_1)
        
        # 策略改进公式: K = R^-1 * B^T * P
        K_i = np.linalg.inv(env.R) @ env.B_lin.T @ P_hat
        
    print(f"收敛后的 P 矩阵:\n{P_hat}")
    print(f"收敛后的 K 矩阵:\n{K_i}")
    
    # 包装一个兼容的 Actor 用于画图
    class DummyActorLinear:
        def __call__(self, x_tensor):
            return -torch.FloatTensor(K_i) @ x_tensor.T

    evaluate_and_plot(env, K_star, DummyActorLinear(), "Module_2_RLS_Linearized")
    return P_hat, K_i


# ========================================================
# 网络结构定义 (通用函数逼近器)
# ========================================================
class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # 参数线性化网络 (LIP)。只使用纯二次型输入。
        # 它的优点是梯度完美平滑，非常适合评估代价函数 V(x)。
        self.fc = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        phi = torch.cat([x1**2, x1*x2, x2**2], dim=1)
        return self.fc(phi)

class QNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super().__init__()
        # 对于 Q 函数，因为它同时耦合了状态和动作，
        # 在非线性环境中极其复杂，故使用多层感知机 (MLP) 代替理论中的 LIP。
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, xu):
        return self.net(xu)

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim, bias=False),
            # 【安全护栏】：最后的 Tanh 用于归一化输出到 [-1, 1]
            nn.Tanh()
        )
        # 【物理限幅】：乘以 15.0，表示物理电机的最大允许扭矩。
        # 这一步防止了纯数据驱动时产生的数学奇点导致力矩输出无穷大。
        self.max_torque = 15.0
        
    def forward(self, x):
        return self.net(x) * self.max_torque


# ========================================================
# 模块三：HDP Actor-Critic
# 真正解决非线性系统的深度强化学习模块
# ========================================================
def module3_hdp(env: VrabieLewisEnv, K_star):
    print("\n--- 模块三：HDP Actor-Critic (严格遵循理论 PI) ---")
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    critic_i = CriticNet(state_dim)
    actor_i = ActorNet(state_dim, action_dim)
    
    # 【预训练 Actor】：论文要求 PI 的初始策略必须是“Admissible (容许的/使系统稳定的)”。
    # 我们用 LQR 算出的次优初始权重 [0, 4] 预热 Actor。
    opt_a0 = optim.Adam(actor_i.parameters(), lr=0.005)
    K_init_t = torch.FloatTensor([[0.0, 4.0]])
    for _ in range(500):
        x = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
        opt_a0.zero_grad()
        nn.MSELoss()(actor_i(x), -x @ K_init_t.T).backward()
        opt_a0.step()
        
    Q_t = torch.FloatTensor(env.Q)
    R_t = torch.FloatTensor(env.R)
    gamma = 0.99 
    
    # 策略迭代外循环
    for i in range(10):
        # 冻结上一代的网络，用于构建目标 (Target)
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.005)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.002)
        
        # 步骤 1. 策略评估 (Policy Evaluation)
        for _ in range(400): # 跑足够多的轮数，解出不动点方程
            x_batch = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            with torch.no_grad():
                u_i = actor_i(x_batch) # 获取当前冻结策略的动作
                # 穿透物理模型预测下一时刻状态
                x_next_batch = x_batch + torch_vrabie_dynamics(x_batch, u_i) * env.dt
                # 计算物理代价 r
                cost_batch = (torch.sum((x_batch @ Q_t) * x_batch, dim=1, keepdim=True) + \
                              torch.sum((u_i @ R_t) * u_i, dim=1, keepdim=True)) * env.dt
                
                # 【自举 (Bootstrapping)】: 目标值 V_target = r + gamma * V_new(x_next)
                # 使用 .detach() 斩断梯度，防止梯度在这个递归公式里无限循环
                target_v = cost_batch + gamma * critic_i_plus_1(x_next_batch).detach()
                
            loss_c = nn.MSELoss()(critic_i_plus_1(x_batch), target_v)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            
        # 步骤 2. 策略改进 (Policy Improvement)
        for _ in range(400):
            x_batch = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_batch)
            x_next_pred = x_batch + torch_vrabie_dynamics(x_batch, u_pred) * env.dt
            
            # 【物理惩罚项】: 显式计算 u^T * R * u。
            # 这是 M3 极其稳定的原因。任何极端的动作都会立刻在这里引发巨大的数学惩罚。
            cost_u = torch.sum((u_pred @ R_t) * u_pred, dim=1, keepdim=True) * env.dt
            
            # 最小化未来总代价
            loss_a = (cost_u + gamma * critic_i_plus_1(x_next_pred)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            
        critic_i = copy.deepcopy(critic_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | Critic Loss: {loss_c.item():.4f} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_3_HDP")
    return actor_i


# ========================================================
# 模块四：ADHDP (彻底免模型的 Q-Learning)
# ========================================================
def module4_adhdp(env: VrabieLewisEnv, K_star):
    print("\n--- 模块四：ADHDP (严格 Q-Learning 式外层迭代) ---")
    state_dim, action_dim = env.state_dim, env.action_dim
    
    q_net_i = QNet(state_dim, action_dim)
    actor_i = ActorNet(state_dim, action_dim)
    
    opt_a0 = optim.Adam(actor_i.parameters(), lr=0.005)
    K_init_t = torch.FloatTensor([[0.0, 4.0]])
    for _ in range(500):
        x = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
        opt_a0.zero_grad()
        nn.MSELoss()(actor_i(x), -x @ K_init_t.T).backward()
        opt_a0.step()
        
    gamma = 0.99
    
    for i in range(10):
        q_net_i_plus_1 = copy.deepcopy(q_net_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_q = optim.Adam(q_net_i_plus_1.parameters(), lr=0.005)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.002)
        
        # 步骤 1. Q 函数评估
        for _ in range(400):
            states, actions, next_states, costs = [], [], [], []
            # 【真实环境交互阶段】
            for _ in range(128):
                x = np.random.uniform([[-2.0], [-2.0]], [[2.0], [2.0]])
                env.x = x
                
                # 【探测噪声 (Probing Noise)】
                # 遵循理论要求: 必须在当前已知策略上叠加噪声，而不是乱探索 (On-policy)
                with torch.no_grad():
                    x_tensor_single = torch.FloatTensor(x.T)
                    u_base = actor_i(x_tensor_single).numpy().T 
                
                # 叠加白噪声 n_k，并在物理限幅内截断
                u = u_base + np.random.normal(0, 1.0, (action_dim, 1))
                u = np.clip(u, -15.0, 15.0) 
                
                x_next, cost = env.step(u) # 丢进真实物理引擎产生反馈
                states.append(x.flatten())
                actions.append(u.flatten())
                next_states.append(x_next.flatten())
                costs.append([cost])
                
            x_t = torch.FloatTensor(np.array(states))
            u_t = torch.FloatTensor(np.array(actions))
            xn_t = torch.FloatTensor(np.array(next_states))
            r_t = torch.FloatTensor(np.array(costs)) 
            
            with torch.no_grad():
                # 算出下一状态的动作: u_next = actor(x_next)
                u_next_i = actor_i(xn_t) 
                # 算出 Target Q = r + gamma * Q(x_next, u_next)
                q_next_i_plus_1 = q_net_i_plus_1(torch.cat([xn_t, u_next_i], dim=1))
                target_q = r_t + gamma * q_next_i_plus_1.detach() # 自举评估
                
            q_pred = q_net_i_plus_1(torch.cat([x_t, u_t], dim=1))
            loss_q = nn.MSELoss()(q_pred, target_q)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            
        # 步骤 2. Actor 改进
        for _ in range(400):
            x_t = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_t)
            
            # 【致命的数学盲区】: 注意这里没有任何物理公式惩罚。
            # Actor 完全凭借 Q 网络(一个黑盒MLP) 给出的评分来优化自己。
            # 极容易陷入 "OOD 动作剥削"，导致最后结果发散。
            loss_a = q_net_i_plus_1(torch.cat([x_t, u_pred], dim=1)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            
        q_net_i = copy.deepcopy(q_net_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | Q Loss: {loss_q.item():.4f} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_4_ADHDP")
    return actor_i


# ========================================================
# 模块五：连续时间积分强化学习 (IRL)
# 专为连续动态系统设计，消灭导数带来的传感器噪声
# ========================================================
def module5_irl(env: VrabieLewisEnv, K_star):
    print("\n--- 模块五：定积分理论映射 (IRL) ---")
    state_dim, action_dim = env.state_dim, env.action_dim
    
    critic_i = CriticNet(state_dim)
    actor_i = ActorNet(state_dim, action_dim)
    
    opt_a0 = optim.Adam(actor_i.parameters(), lr=0.005)
    K_init_t = torch.FloatTensor([[0.0, 4.0]])
    for _ in range(500):
        x = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
        opt_a0.zero_grad()
        nn.MSELoss()(actor_i(x), -x @ K_init_t.T).backward()
        opt_a0.step()
        
    Q_t = torch.FloatTensor(env.Q)
    R_t = torch.FloatTensor(env.R)
    gamma = 0.99 
    
    # 【积分窗口设置】: 设置 T = 0.5 秒。在现实中，这意味着收集 0.5 秒的传感器数据
    T_window = 0.5 
    steps = int(T_window / env.dt)
    
    for i in range(10):
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.005)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.002)
        
        # 1. 连续代价积分评估
        for _ in range(400):
            x_init = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            
            with torch.no_grad():
                x_traj = x_init.clone()
                integral_cost = torch.zeros(128, 1)
                
                # 【微积分仿真循环】: 模拟系统从 t 运行到 t+T
                for _ in range(steps):
                    u_curr = actor_i(x_traj)
                    cost_curr = torch.sum((x_traj @ Q_t) * x_traj, dim=1, keepdim=True) + \
                                torch.sum((u_curr @ R_t) * u_curr, dim=1, keepdim=True)
                    # 将每一步的瞬间代价乘以时间 dt，累加成总面积 (定积分)
                    integral_cost += cost_curr * env.dt
                    x_dot = torch_vrabie_dynamics(x_traj, u_curr)
                    x_traj = x_traj + x_dot * env.dt
                    
                # 【积分贝尔曼方程 Target】: V(x(t)) = \int_{t}^{t+T} r d_tau + V(x(t+T))
                target_v = integral_cost + gamma * critic_i_plus_1(x_traj).detach()
                
            loss_c = nn.MSELoss()(critic_i_plus_1(x_init), target_v) 
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            
        # 2. 策略更新微调
        for _ in range(400):
            x_init = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_init)
            x_dot_pred = torch_vrabie_dynamics(x_init, u_pred)
            x_next_step = x_init + x_dot_pred * env.dt
            cost_u = torch.sum((u_pred @ R_t) * u_pred, dim=1, keepdim=True) * env.dt
            
            loss_a = (cost_u + gamma * critic_i_plus_1(x_next_step)).mean() 
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            
        critic_i = copy.deepcopy(critic_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | IRL Critic: {loss_c.item():.4f} | Actor: {loss_a.item():.4f}")
            
    evaluate_and_plot(env, K_star, actor_i, "Module_5_IRL")
    return actor_i

if __name__ == "__main__":
    env = VrabieLewisEnv(dt=0.05)
    
    print("="*60)
    print("Lewis (2009) ADP: Strict Policy Iteration Formula Translation")
    print("环境: Vrabie-Lewis Benchmark")
    print("="*60)
    
    P_star, K_star = module1_offline_lqr(env)
    
    K_init = np.array([[0.0, 4.0]])
    P_hat, K_hat = module2_online_rls_policy_iteration(env, K_init, K_star)
    
    actor_hdp = module3_hdp(env, K_star)
    actor_adhdp = module4_adhdp(env, K_star)
    actor_irl = module5_irl(env, K_star)
    
    print("\n[全部模块] 依托论文最严格的原汁原味迭代范式公式 (i => i+1) 执行完毕！")