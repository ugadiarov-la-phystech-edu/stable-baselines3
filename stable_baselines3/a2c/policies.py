# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
    register_policy, ActorCriticCSWMPolicy,
)

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
CSWMPolicy = ActorCriticCSWMPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
register_policy("CSWMPolicy", ActorCriticCSWMPolicy)
