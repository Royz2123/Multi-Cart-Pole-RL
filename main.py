import gym

ENVIROMENTS = [
    'CartPole-v0',
    'multi_cart:multi-cart-v0'
]


def main(env_name):
    env = gym.make(env_name)
    for i_episode in range(50):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main(ENVIROMENTS[-1])
