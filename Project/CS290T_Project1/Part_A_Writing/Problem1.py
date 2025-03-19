# 定义状态、动作和参数
states = ['s1', 's2', 's3']
actions = ['a1', 'a2']
gamma = 0.9  # 折扣因子

# 转移概率 P(s'|s, a)
# 格式: P[s][a][s'] 表示 P(s'|s, a)
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
        'a1': {'s1': 0.0, 's2': 0.0, 's3': 0.0},  # 终止状态
        'a2': {'s1': 0.0, 's2': 0.0, 's3': 0.0}
    }
}

# 奖励函数 R(s, a)
R = {
    's1': {'a1': 1, 'a2': 2},
    's2': {'a1': 3, 'a2': 0},
    's3': {'a1': 0, 'a2': 0}
}

# 第一问的策略评估结果（1 次迭代）
V = {
    's1': 1.5,
    's2': 1.5,
    's3': 0.0
}

# 计算 Q(s, a)
Q = {}
for s in states:
    Q[s] = {}
    for a in actions:
        # Q(s, a) = R(s, a) + gamma * sum(P(s'|s, a) * V(s'))
        expected_future_value = sum(P[s][a][s_next] * V[s_next] for s_next in states)
        Q[s][a] = R[s][a] + gamma * expected_future_value

# 打印 Q(s, a)
print("Q(s, a) 值：")
for s in states:
    for a in actions:
        print(f"Q({s}, {a}) = {Q[s][a]:.4f}")

# 策略改进：选择 Q(s, a) 最大的动作
new_policy = {}
for s in states:
    if s == 's3':  # 终止状态无需选择动作
        new_policy[s] = None
        continue
    # 找到最大的 Q 值和对应的动作
    best_action = max(actions, key=lambda a: Q[s][a])
    new_policy[s] = best_action

# 打印新策略
print("\n新策略 pi'(s)：")
for s in states:
    if s != 's3':
        print(f"pi'({s}) = {new_policy[s]}")
    else:
        print(f"pi'({s}) = None (终止状态)")