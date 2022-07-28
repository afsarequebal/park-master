import warnings
import park


def run_env_with_random_agent(env_name: object, seed: object) -> object:
    # suppress unittest from throwing weird warnings
    warnings.simplefilter('ignore', category=ImportWarning)

    env = park.make(env_name)
    env.seed(seed)

    obs = env.reset()
    done = False
    actions = []
    observations = []
    while not done:

        act = env.action_space.sample()
        obs, reward, done, info = env.step(act)
        actions.append(act)
        observations.append(obs)
    print(actions)
    print(observations)


