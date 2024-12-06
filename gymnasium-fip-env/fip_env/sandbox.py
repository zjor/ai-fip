import imageio.v3 as iio
import gymnasium
import numpy as np

if __name__ == '__main__':
    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode="rgb_array")
    env.reset()
    pixels = env.render()
    iio.imwrite(f"output.png", pixels)

