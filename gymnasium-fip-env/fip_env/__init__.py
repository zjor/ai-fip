from gymnasium.envs.registration import register

register(id="fip_env/GridWorld-v0", entry_point="fip_env.envs:GridWorldEnv")
register(id="fip_env/BallCatcher-v0", entry_point="fip_env.envs:BallCatcherEnv")
register(id="fip_env/FlywheelInvertedPendulum-v0", entry_point="fip_env.envs:FlywheelInvertedPendulumEnv")
