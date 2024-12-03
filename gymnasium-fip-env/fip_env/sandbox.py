import gymnasium

if __name__ == '__main__':
    env = gymnasium.make('fip_env/GridWorld-v0')
    env.reset()
    print(env.action_space.sample())