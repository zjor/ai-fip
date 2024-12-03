from gymnasium.envs.registration import register

register(
    id="fip_env/GridWorld-v0",
    entry_point="fip_env.envs:GridWorldEnv",
)
