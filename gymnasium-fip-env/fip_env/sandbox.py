import imageio.v3 as iio
import gymnasium
import numpy as np

if __name__ == '__main__':
    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode="human")
    env.reset()
    for i in range(220):
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            break
        # pixels = env.render()
        # iio.imwrite(f"output-{i}.png", pixels)
