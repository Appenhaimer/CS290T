# 定义状态、动作和参数
states = ['s1', 's2', 's3']
actions = ['a1', 'a2']
gamma = 0.9  # 折扣因子

# 转移概率 P(s'|s, a)
P = {
    's1': {
        'a1': {'s1': 0.0, 's2': 0.5, 's3': 0.5},
        'a2': {'s1': 0.0, 's2': 0.7, 's3': 0.3}
    },
    's2': {
        'a1': {'s1': 0.6, 's2': 0.0, 's3': 0.4},
        'a2': {'s1': 0.8, 's2': 0.0, 's3': 0.2}
    },
    's3': {
        'a1': {'s1': 0.0, 's2': 0.0, 's3': 0.0},
        'a2': {'s1': 0.0, 's2': 0.0, 's3': 0.0}
    }
}

# 奖励函数 R(s, a)
R = {
    's1': {'a1': 1, 'a2': 2},
    's2': {'a1': 3, 'a2': 0},
    's3': {'a1': 0, 'a2': 0}
}

# 初始价值函数 V(s) = 0
V = {
    's1': 0.0,
    's2': 0.0,
    's3': 0.0
}

# 价值迭代：10 轮
num_iterations = 1
for iteration in range(num_iterations):
    new_V = V.copy()  # 存储新值
    for s in states:
        # 计算每个动作的 Q(s, a)
        Q_values = []
        for a in actions:
            expected_future_value = sum(P[s][a][s_next] * V[s_next] for s_next in states)
            Q = R[s][a] + gamma * expected_future_value
            Q_values.append(Q)
        # 更新 V(s) = max_a Q(s, a)
        new_V[s] = max(Q_values)
    V = new_V  # 更新 V
    print(f"第 {iteration + 1} 轮价值迭代结果：")
    print(f"V(s1) = {V['s1']:.4f}, V(s2) = {V['s2']:.4f}, V(s3) = {V['s3']:.4f}")

# 基于最终 V(s) 计算策略
policy = {}
for s in states:
    if s == 's3':  # 终止状态
        policy[s] = None
        continue
    # 计算每个动作的 Q(s, a)
    Q_values = {}
    for a in actions:
        expected_future_value = sum(P[s][a][s_next] * V[s_next] for s_next in states)
        Q_values[a] = R[s][a] + gamma * expected_future_value
    # 选择最优动作
    best_action = max(Q_values, key=Q_values.get)
    policy[s] = best_action

# 打印最终策略
print("\n最终策略 pi(s)：")
for s in states:
    if s != 's3':
        print(f"pi({s}) = {policy[s]}")
    else:
        print(f"pi({s}) = None (终止状态)")