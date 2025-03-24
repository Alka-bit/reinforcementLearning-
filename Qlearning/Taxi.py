import random
import gym
import numpy as np

env = gym.make('Taxi-v3')

alpha = 0.9  #learning rate, to which rate does new info overrides old info
gamma = 0.95  #discount factor, how important future rewards are, gamma = 1 means long term reward is as important as immediate reward
epsilon = 1.0   #randomeness, exploration rate, use qtable  
epsilon_Decay = 0.9995
minEpsilon = 0.01
epochs = 100000
maxSteps = 100


qTable = np.zeros((env.observation_space.n, env.action_space.n))

def chooseAction(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else :
        return np.argmax(qTable[state, :])

for epoch in range(epochs):
    state, _ = env.reset()

    done = False

    for step in range(maxSteps):
        action = chooseAction(state)

        nextState, reward, done, truncated, info = env.step(action)

        oldValue = qTable[state, action]
        nextMax = np.max(qTable[nextState , :])

        qTable[state, action] = (1 - alpha) * oldValue + alpha * (reward + gamma * nextMax)

        state = nextState

        if done or truncated:
            break

    epsilon = max(minEpsilon, epsilon*epsilon_Decay)


env = gym.make('Taxi-v3', render_mode = 'human')

for epoch in range(5):
    state, _ = env.reset()
    done= False

    print('Epochs', epoch)

    for step in range(maxSteps):
        env.render()
        action = np.argmax(qTable[state, :])
        nextState, reward, done, truncated, info = env.step(action)
        
        state = nextState

        if done or truncated:
            env.render()
            print('Finished epochs', epoch, 'with', reward)
            break

env.close()
