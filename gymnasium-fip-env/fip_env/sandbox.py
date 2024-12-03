import imageio.v3 as iio
import gymnasium
import numpy as np

if __name__ == '__main__':
    # env = gymnasium.make('fip_env/GridWorld-v0', render_mode="rgb_array")
    # env.reset()
    # for i in range(5):
    #     env.step(env.action_space.sample())
    #     pixels = env.render()
    #     iio.imwrite(f"output-{i}.png", pixels)

    s = gymnasium.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
    print(s.sample())