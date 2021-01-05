from gym.envs.registration import register

register(
    id='multi-cart-v0',
    entry_point='multi_cart.envs:MultiCartPoleEnv',
)
