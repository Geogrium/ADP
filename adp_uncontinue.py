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
# 全局设置
# ========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ========================================================
# 模块零：上帝视角 —— 纯离散时间线性系统
# ========================================================
class DiscreteEnv:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1
        
        # 离散时间系统矩阵: x_{k+1} = A x_k + B u_k
        self.A = np.array([[0.9, 0.05], [0.1, 0.9]], dtype=np.float32)
        self.B = np.array([[0.0], [0.1]], dtype=np.float32)
        
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.R = np.array([[0.1]], dtype=np.float32)

    def reset(self, x0=None):
        if x0 is None:
            self.x = np.random.uniform(-2.0, 2.0, (self.state_dim, 1))
        else:
            self.x = np.array(x0, dtype=np.float32).reshape(self.state_dim, 1)
        return self.x.copy()

    def calc_cost(self, x, u):
        # 纯离散单步代价: r_k = x_k^T Q x_k + u_k^T R u_k
        return (x.T @ self.Q @ x + u.T @ self.R @ u).item()

    def step(self, u):
        u = np.array(u).reshape(self.action_dim, 1)
        cost = self.calc_cost(self.x, u)
        
        # 严格的离散状态转移方程，不再有 dt 参与
        x_next = self.A @ self.x + self.B @ u
        self.x = x_next
        
        return self.x.copy(), cost

# 供 PyTorch 自动求导的离散动力学
def torch_discrete_dynamics(x, u, A_t, B_t):
    return x @ A_t.T + u @ B_t.T


# ========================================================
# 验证与可视化工具
# ========================================================
def evaluate_and_plot(env: DiscreteEnv, K_star, actor_net, module_name=""):
    x0 = np.array([[2.0], [-2.0]]) 
    steps = 50 # 离散系统通常步数较少
    
    # 1. 运行 DARE 最优 LQR 策略
    env.reset(x0)
    lqr_traj = [env.x.copy()]
    lqr_cost = 0.0
    for _ in range(steps):
        u = -K_star @ env.x
        x_next, cost = env.step(u)
        lqr_traj.append(x_next)
        lqr_cost += cost
        
    # 2. 运行 Actor 策略
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
        
    lqr_traj = np.array(lqr_traj).squeeze()
    actor_traj = np.array(actor_traj).squeeze()
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"[{module_name}] LQR Cost:{lqr_cost:.2f} | Actor Cost:{actor_cost:.2f}")
    
    plt.subplot(1, 2, 1)
    plt.plot(lqr_traj[:, 0], label='LQR x1', linestyle='--')
    if actor_net is not None:
        plt.plot(actor_traj[:, 0], label='Actor x1', alpha=0.9, linestyle='-.')
    plt.title("State x1 Trajectory (Discrete Steps)")
    plt.xlabel("Step k")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(lqr_traj[:, 0], lqr_traj[:, 1], label='LQR Phase', linestyle='--', marker='o', markersize=3)
    if actor_net is not None:
        plt.plot(actor_traj[:, 0], actor_traj[:, 1], label='Actor Phase', marker='x', markersize=3, linestyle='-.')
    plt.title("Phase Portrait (x1 vs x2)")
    plt.legend()
    plt.grid()
    
    filename = f"{module_name.replace(' ', '_')}_validation.png"
    plt.savefig(filename)
    plt.close()
    print(f"[{module_name}] 验证图生成: LQR={lqr_cost:.2f} | Actor={actor_cost:.2f}")


# ========================================================
# 模块一：离线 LQR (解离散代数黎卡提方程 DARE)
# ========================================================
def module1_offline_discrete_lqr(env: DiscreteEnv):
    # 严格调用 solve_discrete_are
    P = scipy.linalg.solve_discrete_are(env.A, env.B, env.Q, env.R)
    # 离散系统的最优增益公式
    K = np.linalg.inv(env.R + env.B.T @ P @ env.B) @ (env.B.T @ P @ env.A)
    print("\n--- 模块一：离线离散 LQR (DARE) ---")
    print(f"解得的离散 P 矩阵:\n{P}")
    print(f"解得的离散 K 矩阵:\n{K}")
    return P, K


# ========================================================
# 模块二：在线 Policy Iteration (基于离散闭式最小二乘)
# [论文推导]: \phi(x_k) - \gamma \phi(x_{k+1}) = r_k
# ========================================================
def phi_rls(x):
    x1, x2 = x[0,0], x[1,0]
    return np.array([[x1**2], [2*x1*x2], [x2**2]])

def get_P_from_W(W):
    w1, w2, w3 = W.flatten()
    return np.array([[w1, w2], [w2, w3]])

def module2_online_rls_discrete_pi(env: DiscreteEnv, K_init, K_star):
    print("\n--- 模块二：在线离散 Policy Iteration (闭式最小二乘) ---")
    K_i = np.array(K_init, dtype=np.float32)
    gamma = 1.0 # 对于无折扣 LQR 等价，gamma 为 1
    
    for i in range(1, 11):
        X_mat = []
        Y_mat = []
        for _ in range(500):
            env.x = np.random.uniform(-2.0, 2.0, (2, 1))
            x_k = env.x.copy()
            u_k = -K_i @ x_k
            
            x_next, cost_k = env.step(u_k)
            
            # 离散差分方程特征映射
            delta_phi = phi_rls(x_k) - gamma * phi_rls(x_next)
            
            X_mat.append(delta_phi.T)
            Y_mat.append(cost_k) 
            
        X_mat = np.vstack(X_mat)
        Y_mat = np.array(Y_mat).reshape(-1, 1)
        
        W_i_plus_1 = np.linalg.pinv(X_mat) @ Y_mat
        P_hat = get_P_from_W(W_i_plus_1)
        
        # 离散系统策略改进公式
        K_i = np.linalg.inv(env.R + env.B.T @ P_hat @ env.B) @ (env.B.T @ P_hat @ env.A)
        
    print(f"收敛后的离散 P 矩阵:\n{P_hat}")
    print(f"收敛后的离散 K 矩阵:\n{K_i}")
    
    class DummyActorLinear:
        def __call__(self, x_tensor):
            return -torch.FloatTensor(K_i) @ x_tensor.T

    evaluate_and_plot(env, K_star, DummyActorLinear(), "Module_2_Discrete_RLS")
    return P_hat, K_i


# ========================================================
# 离散网络结构定义
# ========================================================
class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用线性参数化(LIP)适配线性系统，加速收敛
        self.fc = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        phi = torch.cat([x1**2, x1*x2, x2**2], dim=1)
        return self.fc(phi)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 1, bias=False)
        
    def forward(self, xu):
        x1, x2, u = xu[:, 0:1], xu[:, 1:2], xu[:, 2:3]
        phi = torch.cat([x1**2, x1*x2, x2**2, x1*u, x2*u, u**2], dim=1)
        return self.fc(phi)

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        
    def forward(self, x):
        return self.fc(x)


# ========================================================
# 模块三：离散 HDP Actor-Critic
# [论文方程 Eq. 33]: V_{i+1}(x_k) = r_k + \gamma V_{i+1}(x_{k+1})
# ========================================================
def module3_discrete_hdp(env: DiscreteEnv, K_star):
    print("\n--- 模块三：离散 HDP Actor-Critic (纯正的离散 PI) ---")
    critic_i = CriticNet()
    actor_i = ActorNet()
    
    # 稳定初始策略 K=[0,0] 在此系统中是 Admissible 的
    torch.nn.init.zeros_(actor_i.fc.weight)
        
    Q_t = torch.FloatTensor(env.Q)
    R_t = torch.FloatTensor(env.R)
    A_t = torch.FloatTensor(env.A)
    B_t = torch.FloatTensor(env.B)
    gamma = 1.0 
    
    for i in range(10):
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.01)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.01)
        
        # 1. 策略评估 (Bootstrapping 自举求解方程)
        for _ in range(400):
            x_k = torch.empty(128, 2).uniform_(-2.0, 2.0)
            with torch.no_grad():
                u_k = actor_i(x_k) 
                x_next = torch_discrete_dynamics(x_k, u_k, A_t, B_t)
                cost_k = torch.sum((x_k @ Q_t) * x_k, dim=1, keepdim=True) + \
                         torch.sum((u_k @ R_t) * u_k, dim=1, keepdim=True)
                
                target_v = cost_k + gamma * critic_i_plus_1(x_next).detach()
                
            loss_c = nn.MSELoss()(critic_i_plus_1(x_k), target_v)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
            
        # 2. 策略改进 [论文方程 Eq. 65]
        for _ in range(400):
            x_k = torch.empty(128, 2).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_k)
            x_next_pred = torch_discrete_dynamics(x_k, u_pred, A_t, B_t)
            cost_u = torch.sum((u_pred @ R_t) * u_pred, dim=1, keepdim=True)
            
            loss_a = (cost_u + gamma * critic_i_plus_1(x_next_pred)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            
        critic_i = copy.deepcopy(critic_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | Critic Loss: {loss_c.item():.4f} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_3_Discrete_HDP")
    return actor_i


# ========================================================
# 模块四：离散 ADHDP (Q-Learning)
# ========================================================
def module4_discrete_adhdp(env: DiscreteEnv, K_star):
    print("\n--- 模块四：离散 ADHDP (离散 Q-Learning) ---")
    q_net_i = QNet()
    actor_i = ActorNet()
    
    torch.nn.init.zeros_(actor_i.fc.weight)
    gamma = 1.0
    
    for i in range(10):
        q_net_i_plus_1 = copy.deepcopy(q_net_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_q = optim.Adam(q_net_i_plus_1.parameters(), lr=0.01)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.01)
        
        # Q Evaluation
        for _ in range(400):
            states, actions, next_states, costs = [], [], [], []
            for _ in range(128):
                env.x = np.random.uniform(-2.0, 2.0, (2, 1))
                x_k = env.x.copy()
                
                with torch.no_grad():
                    u_base = actor_i(torch.FloatTensor(x_k.T)).numpy().T 
                
                u_k = u_base + np.random.normal(0, 0.5, (1, 1))
                x_next, cost_k = env.step(u_k)
                
                states.append(x_k.flatten())
                actions.append(u_k.flatten())
                next_states.append(x_next.flatten())
                costs.append([cost_k])
                
            x_t = torch.FloatTensor(np.array(states))
            u_t = torch.FloatTensor(np.array(actions))
            xn_t = torch.FloatTensor(np.array(next_states))
            r_t = torch.FloatTensor(np.array(costs)) 
            
            with torch.no_grad():
                u_next_i = actor_i(xn_t) 
                q_next_i_plus_1 = q_net_i_plus_1(torch.cat([xn_t, u_next_i], dim=1))
                target_q = r_t + gamma * q_next_i_plus_1.detach()
                
            q_pred = q_net_i_plus_1(torch.cat([x_t, u_t], dim=1))
            loss_q = nn.MSELoss()(q_pred, target_q)
            opt_q.zero_grad(); loss_q.backward(); opt_q.step()
            
        # Q Improvement
        for _ in range(400):
            x_k = torch.empty(128, 2).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_k)
            loss_a = q_net_i_plus_1(torch.cat([x_k, u_pred], dim=1)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            
        q_net_i = copy.deepcopy(q_net_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | Q Loss: {loss_q.item():.4f} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_4_Discrete_ADHDP")
    return actor_i


# ========================================================
# 模块五：离散 Value Iteration (VI)
# [替代 IRL] 论文中强调的离散情况下的另一种基础迭代方法
# [论文方程 Eq. 35]: V_{i+1}(x_k) = \min_u [r_k + \gamma V_i(x_{k+1})]
# ========================================================
# ========================================================
# 模块五：离线 Value Iteration (VI) - 修改为 50 次迭代
# ========================================================
def module5_discrete_vi(env: DiscreteEnv, K_star):
    print("\n--- 模块五：离线离散 Value Iteration (VI) ---")
    critic_i = CriticNet() # 初始价值函数 V_0
    actor_i = ActorNet()   # 初始策略
    
    Q_t = torch.FloatTensor(env.Q)
    R_t = torch.FloatTensor(env.R)
    A_t = torch.FloatTensor(env.A)
    B_t = torch.FloatTensor(env.B)
    gamma = 1.0 
    
    # 将迭代次数从 15 提升至 50
    # 在离散线性系统中，这能保证 V_i 收敛到 DARE 的固定点解 P
    for i in range(50):
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.01)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.01)
        
        # 内部优化循环
        for _ in range(400):
            x_k = torch.empty(128, 2).uniform_(-2.0, 2.0)
            
            # 1. 更新 Actor: 寻找使 [r + gamma * V_i(x_next)] 最小的 u
            # 注意这里使用的是上一轮冻结的 critic_i
            u_pred = actor_i_plus_1(x_k)
            x_next_pred = torch_discrete_dynamics(x_k, u_pred, A_t, B_t)
            cost_u = torch.sum((u_pred @ R_t) * u_pred, dim=1, keepdim=True)
            
            loss_a = (cost_u + gamma * critic_i(x_next_pred).detach()).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            
            # 2. 更新 Critic: 逼近贝尔曼最优方程 V_{i+1} = min_u [r + gamma * V_i]
            u_opt = actor_i_plus_1(x_k).detach()
            x_next_opt = torch_discrete_dynamics(x_k, u_opt, A_t, B_t)
            cost_k = torch.sum((x_k @ Q_t) * x_k, dim=1, keepdim=True) + \
                     torch.sum((u_opt @ R_t) * u_opt, dim=1, keepdim=True)
                     
            target_v = cost_k + gamma * critic_i(x_next_opt).detach()
            loss_c = nn.MSELoss()(critic_i_plus_1(x_k), target_v)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
            
        critic_i = copy.deepcopy(critic_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        
        # 每 10 次打印一次进度
        if (i+1) % 10 == 0 or i == 0:
            print(f"VI Iteration {i+1}/50 | Critic Loss: {loss_c.item():.6e} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_5_Discrete_VI_50iters")
    return actor_i

if __name__ == "__main__":
    env = DiscreteEnv()
    
    print("="*60)
    print("Lewis (2009) ADP: Strict DISCRETE-TIME Translation")
    print("============================================================")
    
    P_star, K_star = module1_offline_discrete_lqr(env)
    
    K_init = np.array([[0.0, 0.0]])
    P_hat, K_hat = module2_online_rls_discrete_pi(env, K_init, K_star)
    
    actor_hdp = module3_discrete_hdp(env, K_star)
    actor_adhdp = module4_discrete_adhdp(env, K_star)
    actor_vi = module5_discrete_vi(env, K_star)
    
    print("\n[全部模块] 纯离散时间差分方程复现执行完毕！")