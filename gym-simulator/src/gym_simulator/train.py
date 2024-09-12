from gym_simulator.releaser_env import CloudSimReleaserEnv


def main():
    env = CloudSimReleaserEnv()
    obs, _ = env.reset()
    for _ in range(1000):
        action = 1 if obs[0] > 20 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}")
        if terminated or truncated:
            env.close()
            break


if __name__ == "__main__":
    main()
