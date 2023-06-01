# Q 값을 초기화
Q = np.zeros([
    env.observation_space.n, 
    env.action_space.n
    ])

#에피소드 진행
state = env.reset()
done = False

while not done:
    action = rargmax(Q[state, :])

    new_state, reward, done, _ = env.step(action)

    Q[state, action] = reward + np.max(Q[new_state, :])

    state = new_state