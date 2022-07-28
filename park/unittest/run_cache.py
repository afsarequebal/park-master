import warnings
import park
from simpleDqn import Agent
from utils import plotLearning
import numpy as np

def run_env_with_dqn_agent(env_name: object, seed: object) -> object:
    # suppress unittest from throwing weird warnings
    warnings.simplefilter('ignore', category=ImportWarning)

    env = park.make(env_name)
    env.seed(seed)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2,
                  eps_end=0.01, input_dims=[3], lr=0.003)
    scores, eps_history = [], []
    n_turn = 10
    obs = env.reset()
    for i in range(n_turn):
        score = 0
        done = False
        observation = env.reset()
        observation = np.array(observation, dtype=np.float32)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = np.array(observation_, dtype=np.float32)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_turn)]
    filename = 'lunar_lander_2020.png'
    plotLearning(x, scores, eps_history, filename)
