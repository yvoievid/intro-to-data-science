from gymnasium.envs.registration import register

register(
    id="operation_simulation/SafePath-v0",
    entry_point="operation_simulation.envs:SafePath",
)
