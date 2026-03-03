from Agent import Agent
from Preprocessing import stack_frames
from env import make_env
import numpy as np
from collections import deque

env = make_env()
num_actions = env.action_space.n
agent = Agent(num_actions)

episodes = 500
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

scores = deque(maxlen=100)

for episode in range(1, episodes + 1):

    state_raw, _ = env.reset()
    state = stack_frames(state_raw, True)

    total_reward = 0

    while True:
        action = agent.act(state, epsilon)
        next_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = stack_frames(next_raw, False)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode} | Avg Score: {np.mean(scores):.2f} | Epsilon: {epsilon:.3f}")
