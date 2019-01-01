from gym.envs.registration import register

register(
    id='ode-v0',
        entry_point='gym_ode.envs:OdeEnv',
)