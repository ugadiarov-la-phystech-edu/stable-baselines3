import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.cswm import TransitionGNN

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(MultiInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class TransitionModel(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        action_dim = get_action_dim(self.action_space)
        self.network = nn.Sequential(*create_mlp(features_dim + action_dim, features_dim, net_arch, activation_fn))

    def forward(self, states: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        states_actions = th.cat([states, actions], dim=1)
        return self.network(states_actions) + states


class TransitionModelGNN(BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            embedding_dim,
            hidden_dim,
            num_objects,
            ignore_action,
            copy_action,
            use_interactions,
            edge_actions
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.edge_actions = edge_actions
        self.network = TransitionGNN(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                     action_dim=self.action_space.shape[0], num_objects=self.num_objects,
                                     ignore_action=self.ignore_action, copy_action=self.copy_action,
                                     use_interactions=self.use_interactions, edge_actions=self.edge_actions)

    def forward(self, embedding_action_boxes_viz):
        return self.network(embedding_action_boxes_viz)[0] + embedding_action_boxes_viz[0]

    def _update_features_extractor(
            self,
            net_kwargs: Dict[str, Any],
            features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        raise NotImplementedError()

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            normalize_images=self.normalize_images,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_objects=self.num_objects,
            ignore_action=self.ignore_action,
            copy_action=self.copy_action,
            use_interactions=self.use_interactions,
            edge_actions=self.edge_actions
        )


class RewardModel(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        action_dim = get_action_dim(self.action_space)
        self.network = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch, activation_fn))

    def forward(self, states: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        states_actions = th.cat([states, actions], dim=1)
        return self.network(states_actions)


class RewardModelGNN(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        embedding_dim,
        hidden_dim,
        num_objects,
        ignore_action,
        copy_action,
        use_interactions,
        edge_actions
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.edge_actions = edge_actions
        self.gnn = TransitionGNN(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                 action_dim=self.action_space.shape[0], num_objects=self.num_objects,
                                 ignore_action=False, copy_action=True, act_fn='relu', layer_norm=True, num_layers=3,
                                 use_interactions=True, edge_actions=True)
        self.mlp = nn.Linear(self.embedding_dim, 1)

    def forward(self, embedding_action_boxes_viz):
        return self.mlp(self.gnn(embedding_action_boxes_viz)[0].mean(dim=1)).squeeze(dim=1)

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        raise NotImplementedError()

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            normalize_images=self.normalize_images,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_objects=self.num_objects,
            ignore_action=self.ignore_action,
            copy_action=self.copy_action,
            use_interactions=self.use_interactions,
            edge_actions=self.edge_actions
        )


class ValueModel(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            normalize_images=False,
        )

        self.network = nn.Sequential(*create_mlp(features_dim, 1, net_arch, activation_fn))

    def forward(self, states: th.Tensor) -> Tuple[th.Tensor, ...]:
        return self.network(states)


class ValueModelGNN(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        embedding_dim,
        hidden_dim,
        num_objects,
        use_interactions,
    ):
        super().__init__(
            observation_space,
            None,
            features_extractor=None,
            normalize_images=False,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.use_interactions = use_interactions
        self.network = TransitionGNN(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                     action_dim=0, num_objects=self.num_objects,
                                     ignore_action=True, copy_action=False,
                                     use_interactions=self.use_interactions, edge_actions=False,)
        self.mlp = nn.Linear(self.embedding_dim, 1)

    def forward(self, embedding_action_boxes_viz):
        gnn_output = self.network(embedding_action_boxes_viz)[0].mean(dim=1)
        return self.mlp(gnn_output).squeeze(dim=1)

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        raise NotImplementedError()

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            normalize_images=self.normalize_images,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_objects=self.num_objects,
            use_interactions=self.use_interactions,
        )


class TerminationModelGNN(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        embedding_dim,
        hidden_dim,
        num_objects,
        use_interactions,
    ):
        super().__init__(
            observation_space,
            None,
            features_extractor=None,
            normalize_images=False,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.use_interactions = use_interactions
        self.gnn = TransitionGNN(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                 action_dim=0, num_objects=self.num_objects,
                                 ignore_action=True, copy_action=False,
                                 use_interactions=self.use_interactions, edge_actions=False,)
        self.mlp = nn.Linear(self.embedding_dim, 1)

    def forward(self, embedding_action_boxes_viz):
        gnn_output = self.gnn(embedding_action_boxes_viz)[0].mean(dim=1)
        return self.mlp(gnn_output).squeeze(dim=1)

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        raise NotImplementedError()

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            normalize_images=self.normalize_images,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_objects=self.num_objects,
            use_interactions=self.use_interactions,
        )


class ContinuousCriticWM(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Dict[str, List[int]],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 1,
        share_features_extractor: bool = True,
        gamma: float = 0.99,
        depth: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.gamma = gamma
        self.depth = depth
        self.n_critics = n_critics
        self.transition_models = nn.ModuleList()
        self.reward_models = nn.ModuleList()
        self.value_models = nn.ModuleList()
        for _ in range(self.n_critics):
            self.transition_models.append(
                TransitionModel(observation_space, action_space, net_arch['transition_model'], features_dim,
                                activation_fn))
            self.reward_models.append(
                RewardModel(observation_space, action_space, net_arch['reward_model'], features_dim, activation_fn))
            self.value_models.append(
                ValueModel(observation_space, action_space, net_arch['value_model'], features_dim, activation_fn))

        self.share_features_extractor = share_features_extractor

    def forward(self, obs: th.Tensor, actions_init: th.Tensor, actor: Actor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)

        q_values = []
        for critic_id in range(self.n_critics):
            transition_model = self.transition_models[critic_id]
            reward_model = self.reward_models[critic_id]
            value_model = self.value_models[critic_id]
            states = [features]
            actions = [actions_init]
            for _ in range(self.depth):
                next_state = transition_model(states[-1], actions[-1])
                next_action = actor(next_state)
                states.append(next_state)
                actions.append(next_action)

            q_value = value_model(states[self.depth])
            for i in range(self.depth - 1, -1, -1):
                q_value = reward_model(states[i], actions[i]) + self.gamma * q_value
            q_values.append(q_value)

        return tuple(q_values)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor, actor: Actor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs)

        transition_model = self.transition_models[0]
        reward_model = self.reward_models[0]
        value_model = self.value_models[0]
        states = [features]
        actions = [actions]
        for _ in range(self.depth):
            next_state = transition_model(states[-1], actions[-1])
            next_action = actor(next_state)
            states.append(next_state)
            actions.append(next_action)

        q_value = value_model(states[self.depth])
        for i in range(self.depth - 1, -1, -1):
            q_value = reward_model(states[i], actions[i]) + self.gamma * q_value

        return q_value


class ContinuousCriticWM_GNN(BaseModel):
    def __init__(
        self,
        transition_model,
        reward_model,
        value_model,
        termination_model=None,
        n_critics: int = 1,
        gamma: float = 0.99,
        depth: int = 1,
    ):
        super().__init__(
            None,
            None,
            features_extractor=None,
            normalize_images=False,
        )

        self.gamma = gamma
        self.depth = depth
        self.n_critics = n_critics
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.value_models = nn.ModuleList()
        self.value_models.append(value_model)
        self.termination_model = termination_model

        def reinit(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for _ in range(self.n_critics - 1):
            value_model_copy = copy.deepcopy(value_model)
            value_model_copy.apply(reinit)
            self.value_models.append(value_model_copy)

    def forward(self, embedding: th.Tensor, action: th.Tensor, moving_boxes: th.Tensor, actor: Actor) -> Tuple[th.Tensor, ...]:
        q_values = []
        for critic_id in range(self.n_critics):
            transition_model = self.transition_model
            reward_model = self.reward_model
            value_model = self.value_models[critic_id]
            states = [embedding]
            actions = [action]
            for _ in range(self.depth):
                next_state = transition_model([states[-1], actions[-1], moving_boxes, False])
                next_action = actor([next_state, actions[-1], moving_boxes, False])
                states.append(next_state)
                actions.append(next_action)

            q_value = value_model([states[self.depth], actions[-1], moving_boxes, False])
            for i in range(self.depth - 1, -1, -1):
                termination = 0
                if self.termination_model is not None:
                    termination = th.sigmoid(self.termination_model([states[i + 1], actions[-1], moving_boxes, False]))
                q_value = reward_model([states[i], actions[i], moving_boxes, False]) + self.gamma * (1 - termination) * q_value
            q_values.append(q_value)

        return tuple(q_values)

    def q1_forward(self, embedding: th.Tensor, actions: th.Tensor, moving_boxes: th.Tensor, actor: Actor) -> th.Tensor:
        transition_model = self.transition_model
        reward_model = self.reward_model
        value_model = self.value_models[0]
        states = [embedding]
        actions = [actions]
        for _ in range(self.depth):
            next_state = transition_model([states[-1], actions[-1], moving_boxes, False])
            next_action = actor([next_state, actions[-1], moving_boxes, False])
            states.append(next_state)
            actions.append(next_action)

        q_value = value_model([states[self.depth], actions[-1], moving_boxes, False])
        for i in range(self.depth - 1, -1, -1):
            termination = 0
            if self.termination_model is not None:
                termination = th.sigmoid(self.termination_model([states[i + 1], actions[-1], moving_boxes, False]))
            q_value = reward_model([states[i], actions[i], moving_boxes, False]) + self.gamma * (1 - termination) * q_value

        return q_value


class ActorGNN(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        embedding_dim,
        hidden_dim,
        num_objects,
        use_interactions,
    ):
        super(ActorGNN, self).__init__(
            observation_space,
            action_space,
            features_extractor=False,
            normalize_images=False,
            squash_output=True,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.use_interactions = use_interactions

        action_dim = get_action_dim(self.action_space)
        self.latent_pi = TransitionGNN(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                       action_dim=0, num_objects=self.num_objects,
                                       ignore_action=True, copy_action=False,
                                       use_interactions=self.use_interactions, edge_actions=False,)
        last_layer_dim = self.embedding_dim
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                squash_output=self.squash_output,
                embedding_dim=self.embedding_dim,
                num_objects=self.num_objects,
                hidden_dim=self.hidden_dim,
                use_interactions=self.use_interactions,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        raise NotImplementedError()

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError()

    def get_action_dist_params(self, embed_action_boxes_viz) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        latent_pi = self.latent_pi(embed_action_boxes_viz)[0].squeeze(-1).mean(-2)
        mean_actions = self.mu(latent_pi)

        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, embed_action_boxes_viz, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(embed_action_boxes_viz)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, embed_action_boxes_viz) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(embed_action_boxes_viz)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, embed_action_boxes_viz, deterministic: bool = False) -> th.Tensor:
        return self(embed_action_boxes_viz, deterministic)

    def predict(
        self,
        embed_action_boxes_viz,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        with th.no_grad():
            actions = self._predict(embed_action_boxes_viz, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions, state


class SACWMPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, List[int]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        gamma: float = 0.99,
        depth: int = 1,
    ):
        super(SACWMPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        assert share_features_extractor, f'Features extractor must be shared!'

        self.gamma = gamma
        self.depth = depth

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

            net_arch = {'actor': net_arch.copy(), 'critic': {model_name: net_arch.copy() for model_name in
                                                             ('transition_model', 'reward_model', 'value_model')}}

        actor_arch, critic_arch = net_arch['actor'], net_arch['critic']

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "gamma": self.gamma,
                "depth": self.depth,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()

        self.features_extractor_target = self.make_features_extractor()
        self.features_extractor_target.load_state_dict(self.features_extractor.state_dict())
        self.features_extractor_target.train(False)

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor(FlattenExtractor(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.features_extractor.features_dim,), dtype=np.float32)))
        self.actor.optimizer = self.optimizer_class(
            list(self.actor.parameters()) + list(self.features_extractor.parameters()), lr=lr_schedule(1),
            **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # It is OK to use the same features extractor as long as it is an instance of FlattenExtractor
        self.critic_target = self.make_critic(features_extractor=self.actor.features_extractor)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                gamma=self.gamma,
                depth=self.depth,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCriticWM:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCriticWM(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(observation)
        return self.actor(features, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.features_extractor.train(mode)
        self.training = mode

    def extract_features_target(self, obs: th.Tensor) -> th.Tensor:
        features_extractor = self.features_extractor
        self.features_extractor = self.features_extractor_target
        features_target = self.extract_features(obs)
        self.features_extractor = features_extractor
        return features_target


class SACWMGNNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr: float,
        features_extractor,
        actor,
        critic,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(SACWMGNNPolicy, self).__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        self.features_extractor = features_extractor
        self.actor = actor
        self.critic = critic
        self.lr = lr

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=self.lr, **self.optimizer_kwargs)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                squash_output=self.squash_output,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError()

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        raise NotImplementedError()

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCriticWM:
        raise NotImplementedError()

    def forward(self, embed_action_boxes_viz, deterministic: bool = False) -> th.Tensor:
        return self._predict(embed_action_boxes_viz, deterministic=deterministic)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return self.features_extractor(obs.to(self.device) / 255.)

    def _predict(self, embed_action_boxes_viz, deterministic: bool = False) -> th.Tensor:
        return self.actor(embed_action_boxes_viz, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def extract_features_target(self, obs: th.Tensor) -> th.Tensor:
        return self.extract_features(obs)


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
