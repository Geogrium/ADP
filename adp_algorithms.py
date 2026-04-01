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
# 模块零：上帝视角 —— Vrabie-Lewis Benchmark (2009)
# ========================================================
class VrabieLewisEnv:
    def __init__(self, dt=0.05):
        self.state_dim = 2
        self.action_dim = 1
        self.dt = dt
        self.x = np.zeros((self.state_dim, 1))
        
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.R = np.array([[1.0]], dtype=np.float32)
        
        self.A_lin = np.array([[-1.0, 1.0], [-0.5, 4.0]], dtype=np.float32)
        self.B_lin = np.array([[0.0], [3.0]], dtype=np.float32)

    def reset(self, x0=None):
        if x0 is None:
            self.x = np.random.uniform([[-1.5], [-1.5]], [[1.5], [1.5]])
        else:
            self.x = np.array(x0, dtype=np.float32).reshape(self.state_dim, 1)
        return self.x.copy()

    def calc_cost(self, x, u):
        return (x.T @ self.Q @ x + u.T @ self.R @ u).item()

    def optimal_control(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return np.array([[- (np.cos(2*x1) + 2) * x2]], dtype=np.float32)

    def step(self, u):
        u = np.array(u).reshape(self.action_dim, 1)
        cost = self.calc_cost(self.x, u)
        x1, x2 = self.x[0, 0], self.x[1, 0]
        
        dot_x1 = -x1 + x2
        dot_x2 = -0.5*x1 - 0.5*x2*(1 - (np.cos(2*x1)+2)**2) + (np.cos(2*x1)+2)*u[0,0]
        
        self.x[0, 0] += dot_x1 * self.dt
        self.x[1, 0] += dot_x2 * self.dt
        
        return self.x.copy(), cost * self.dt

def torch_vrabie_dynamics(x, u):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    dot_x1 = -x1 + x2
    dot_x2 = -0.5*x1 - 0.5*x2*(1 - (torch.cos(2*x1)+2)**2) + (torch.cos(2*x1)+2)*u
    return torch.cat([dot_x1, dot_x2], dim=1)

def evaluate_and_plot(env: VrabieLewisEnv, K_star, actor_net, module_name=""):
    x0 = np.array([[1.5], [-1.0]]) 
    steps = 100
    
    env.reset(x0)
    opt_traj = [env.x.copy()]
    opt_cost = 0.0
    for _ in range(steps):
        u = env.optimal_control(env.x)
        x_next, cost = env.step(u)
        opt_traj.append(x_next)
        opt_cost += cost
        
    env.reset(x0)
    lqr_traj = [env.x.copy()]
    lqr_cost = 0.0
    for _ in range(steps):
        u = -K_star @ env.x
        x_next, cost = env.step(u)
        lqr_traj.append(x_next)
        lqr_cost += cost
        
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
# 模块一：离线 LQR
# ========================================================
def module1_offline_lqr(env: VrabieLewisEnv):
    P = scipy.linalg.solve_continuous_are(env.A_lin, env.B_lin, env.Q, env.R)
    K = np.linalg.inv(env.R) @ env.B_lin.T @ P
    print("\n--- 模块一：离线 LQR ---")
    print(f"由雅可比矩阵解得的 P 矩阵:\n{P}")
    print(f"局部线性化的 K 矩阵:\n{K}")
    return P, K

# ========================================================
# 模块二：在线 Policy Iteration (基于 RLS / Least Squares 理论映射)
# 公式的严格还原： W^T (phi(x) - phi(x_next)) = r(x, u_i) * dt
# ========================================================
def phi_rls(x):
    x1, x2 = x[0,0], x[1,0]
    return np.array([[x1**2], [2*x1*x2], [x2**2]])

def get_P_from_W(W):
    w1, w2, w3 = W.flatten()
    return np.array([[w1, w2], [w2, w3]])

def module2_online_rls_policy_iteration(env: VrabieLewisEnv, K_init, K_star):
    print("\n--- 模块二：在线 Policy Iteration (基于闭式最小二乘，证明 LQR 等价性) ---")
    K_i = np.array(K_init, dtype=np.float32)
    
    # 严格根据论文执行 i 步的迭代
    for i in range(1, 11):
        X_mat = []
        Y_mat = []
        for _ in range(500):
            env.x = np.random.uniform([[-1.5], [-1.5]], [[1.5], [1.5]])
            x = env.x.copy()
            x1_val, x2_val = x[0,0], x[1,0]
            u_i = -K_i @ x
            
            # 【修正点】: RLS 解析求解必须基于线性化系统才能完美等价 LQR
            # 使用 A_lin 和 B_lin 计算连续时间的导数
            dot_x = env.A_lin @ x + env.B_lin @ u_i
            dot_x1, dot_x2 = dot_x[0,0], dot_x[1,0]
            
            dot_phi = np.array([
                [2 * x1_val * dot_x1],
                [x1_val * dot_x2 + dot_x1 * x2_val],
                [2 * x2_val * dot_x2]
            ])
            
            # 代价函数依然是二次型
            cost_r = env.calc_cost(x, u_i)
            
            X_mat.append(dot_phi.T)
            Y_mat.append(-cost_r) 
            
        X_mat = np.vstack(X_mat)
        Y_mat = np.array(Y_mat).reshape(-1, 1)
        
        W_i_plus_1 = np.linalg.pinv(X_mat) @ Y_mat
        P_hat = get_P_from_W(W_i_plus_1)
        
        # 策略改进：此时使用 B_lin 就是完全正确的
        K_i = np.linalg.inv(env.R) @ env.B_lin.T @ P_hat
        
    print(f"收敛后的 P 矩阵:\n{P_hat}")
    print(f"收敛后的 K 矩阵:\n{K_i}")
    
    class DummyActorLinear:
        def __call__(self, x_tensor):
            return -torch.FloatTensor(K_i) @ x_tensor.T

    # 验证时，它应该逼近 LQR 的性能
    evaluate_and_plot(env, K_star, DummyActorLinear(), "Module_2_RLS_Linearized")
    return P_hat, K_i

# ========================================================
# 网络结构定义
# ========================================================
class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # 完全遵循 Lewis 论文的 LIP (Linear in Parameters) 结构
        # 使用真实的二次基底： [x1^2, x1*x2, x2^2]
        self.fc = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        phi = torch.cat([x1**2, x1*x2, x2**2], dim=1)
        return self.fc(phi)

class QNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super().__init__()
        # 【修正点】: 对于非线性系统，必须使用 MLP 作为 Q 网络
        # 这样才能捕捉到环境里 cos(2*x1) 等复杂的非线性耦合特征
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
            nn.Linear(hidden_dim, action_dim, bias=False)
        )
    def forward(self, x):
        return self.net(x)

# ========================================================
# 模块三：HDP Actor-Critic
# 论文公式： 
# Policy Evaluation: V_{i+1}(x) = r(x, u_i) + V_{i}(x_next)
# Policy Improvement: u_{i+1}(x) = argmin [ r(x, u_{i+1}) + V_{i+1}(x_next) ]
# ========================================================
def module3_hdp(env: VrabieLewisEnv, K_star):
    print("\n--- 模块三：HDP Actor-Critic (严格遵循理论 PI) ---")
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    critic_i = CriticNet(state_dim)
    actor_i = ActorNet(state_dim, action_dim)
    
    # 按照论文：初始化必须是一个 Admissible Policy (稳态控制)
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
    
    # Policy Iteration 外层大循环 (下标 i)
    for i in range(10):
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.005)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.002)
        
        # 步骤 1. 策略评估
        for _ in range(150):
            x_batch = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            with torch.no_grad():
                u_i = actor_i(x_batch) # 给定当前冻结的 policy u_i
                x_next_batch = x_batch + torch_vrabie_dynamics(x_batch, u_i) * env.dt
                cost_batch = (torch.sum((x_batch @ Q_t) * x_batch, dim=1, keepdim=True) + \
                              torch.sum((u_i @ R_t) * u_i, dim=1, keepdim=True)) * env.dt
                # 【目标函数：V_{i+1} 向 r + V_i 锁定逼近，本质即 Target Network!】
                target_v = cost_batch + gamma * critic_i(x_next_batch)
                
            loss_c = nn.MSELoss()(critic_i_plus_1(x_batch), target_v)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            
        # 步骤 2. 策略改进
        for _ in range(150):
            x_batch = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_batch)
            x_next_pred = x_batch + torch_vrabie_dynamics(x_batch, u_pred) * env.dt
            cost_u = torch.sum((u_pred @ R_t) * u_pred, dim=1, keepdim=True) * env.dt
            
            # 【使用全新出炉的 V_{i+1} 指导动作选取微调】
            loss_a = (cost_u + gamma * critic_i_plus_1(x_next_pred)).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            
        # 真正使得 i 步向前推进
        critic_i = copy.deepcopy(critic_i_plus_1)
        actor_i = copy.deepcopy(actor_i_plus_1)
        print(f"PI Iteration {i+1}/10 | Critic Loss: {loss_c.item():.4f} | Actor Loss: {loss_a.item():.4f}")

    evaluate_and_plot(env, K_star, actor_i, "Module_3_HDP")
    return actor_i


# ========================================================
# 模块四：ADHDP (免模型端到端 Q 函数学习)
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
        
        # Q Evaluation
        for _ in range(150):
            states, actions, next_states, costs = [], [], [], []
            for _ in range(128):
                x = np.random.uniform([[-2.0], [-2.0]], [[2.0], [2.0]])
                env.x = x
                u = np.random.uniform(-3.0, 3.0, (action_dim, 1))
                x_next, cost = env.step(u)
                states.append(x.flatten())
                actions.append(u.flatten())
                next_states.append(x_next.flatten())
                costs.append([cost])
                
            x_t = torch.FloatTensor(np.array(states))
            u_t = torch.FloatTensor(np.array(actions))
            xn_t = torch.FloatTensor(np.array(next_states))
            r_t = torch.FloatTensor(np.array(costs)) 
            
            with torch.no_grad():
                # Q_{i+1}(x,u) = r + Q_i(x_next, u_i(x_next))
                u_next_i = actor_i(xn_t) 
                q_next_i = q_net_i(torch.cat([xn_t, u_next_i], dim=1))
                target_q = r_t + gamma * q_next_i
                
            q_pred = q_net_i_plus_1(torch.cat([x_t, u_t], dim=1))
            loss_q = nn.MSELoss()(q_pred, target_q)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            
        # Q Improvement
        for _ in range(150):
            x_t = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            u_pred = actor_i_plus_1(x_t)
            # 最小化新 Q
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
# 模块五：连续时间积分强化学习 (Integral RL)
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
    
    T_window = 0.5 
    steps = int(T_window / env.dt)
    
    for i in range(10):
        critic_i_plus_1 = copy.deepcopy(critic_i)
        actor_i_plus_1 = copy.deepcopy(actor_i)
        
        opt_c = optim.Adam(critic_i_plus_1.parameters(), lr=0.005)
        opt_a = optim.Adam(actor_i_plus_1.parameters(), lr=0.002)
        
        # 1. 连续代价积分模拟
        for _ in range(150):
            x_init = torch.empty(128, state_dim).uniform_(-2.0, 2.0)
            
            with torch.no_grad():
                x_traj = x_init.clone()
                integral_cost = torch.zeros(128, 1)
                for _ in range(steps):
                    u_curr = actor_i(x_traj) # 使用旧策略积分
                    cost_curr = torch.sum((x_traj @ Q_t) * x_traj, dim=1, keepdim=True) + \
                                torch.sum((u_curr @ R_t) * u_curr, dim=1, keepdim=True)
                    integral_cost += cost_curr * env.dt
                    x_dot = torch_vrabie_dynamics(x_traj, u_curr)
                    x_traj = x_traj + x_dot * env.dt
                    
                # Equation: V_{i+1}(x(t)) = \int r d\tau + V_i(x(t+dt))
                target_v = integral_cost + gamma * critic_i(x_traj)
                
            loss_c = nn.MSELoss()(critic_i_plus_1(x_init), target_v) 
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            
        # 2. 策略更新微调
        for _ in range(150):
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
