# dependencies
import gymnasium as gym
import numpy as np
from collections import defaultdict


class BeliefRewardWrapper(gym.Wrapper):
    """
    Belief Reward Shaping (BRS) for Crafter environment.

    Implements Bayesian reward shaping from Marom & Rosman (2018)
    to accelerate learning by providing shaped rewards based on prior
    beliefs about achievement values that decay with experience.

    SB3-compatible: returns legacy 4-tuple (obs, reward, done, info)
    for step() to work with Stable-Baselines3 DQN.
    """

    def __init__(self, env, lambda_param=1000, health_weight=0.1,
                 clip_belief_reward=True, use_clusters=True):
        super().__init__(env)
        self.lambda_param = lambda_param
        self.health_weight = health_weight
        self.clip_belief_reward = clip_belief_reward
        self.use_clusters = use_clusters

        self.prior_means = self._initialize_priors()
        self.achievement_counts = defaultdict(int)
        self.current_belief_means = defaultdict(lambda: 0.0)

        self.belief_clusters = self._define_belief_clusters()
        self.cluster_counts = defaultdict(int)
        self.cluster_belief_means = defaultdict(lambda: 0.0)

        self.unlocked_this_episode = set()
        self.unlocked_ever = set()
        self.prev_health = None

        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_belief_reward = 0.0

        # Preserve metadata for rendering
        if hasattr(env, 'metadata'):
            self.metadata = env.metadata
        else:
            self.metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 30}

    def _initialize_priors(self):
        priors = {}
        # Tier 1
        priors['collect_drink'] = 1.0
        priors['collect_wood'] = 1.0
        priors['collect_sapling'] = 0.8
        priors['wake_up'] = 0.8
        # Tier 2
        priors['place_table'] = 1.5
        priors['eat_cow'] = 1.2
        priors['place_plant'] = 1.0
        priors['defeat_zombie'] = 1.5
        # Tier 3
        priors['make_wood_pickaxe'] = 2.0
        priors['make_wood_sword'] = 1.8
        priors['collect_stone'] = 1.5
        # Tier 4
        priors['make_stone_pickaxe'] = 2.5
        priors['make_stone_sword'] = 2.3
        priors['collect_coal'] = 2.0
        priors['place_stone'] = 1.0
        priors['defeat_skeleton'] = 2.5
        # Tier 5
        priors['place_furnace'] = 3.0
        priors['collect_iron'] = 3.5
        # Tier 6
        priors['make_iron_pickaxe'] = 4.0
        priors['make_iron_sword'] = 4.0
        # Tier 7
        priors['collect_diamond'] = 5.0
        priors['eat_plant'] = 3.5
        return priors

    def _define_belief_clusters(self):
        clusters = {
            'survival_cluster': ['collect_drink', 'eat_cow', 'eat_plant', 'wake_up'],
            'basic_resources_cluster': ['collect_wood', 'collect_sapling', 'place_plant'],
            'stone_tier_cluster': ['collect_stone', 'place_table', 'make_wood_pickaxe',
                                   'make_wood_sword', 'place_stone'],
            'advanced_resources_cluster': ['collect_coal', 'collect_iron', 'collect_diamond'],
            'iron_tier_cluster': ['place_furnace', 'make_stone_pickaxe', 'make_stone_sword',
                                  'make_iron_pickaxe', 'make_iron_sword'],
            'combat_cluster': ['defeat_zombie', 'defeat_skeleton']
        }
        return clusters

    def reset(self, **kwargs):
        """
        Reset wrapper - expects Gymnasium format from wrapped env.
        Returns (obs, info) tuple for modern SB3/Gymnasium compatibility.
        """
        result = self.env.reset(**kwargs)
        # The wrapped env (GymV21CompatibilityV0) returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        self.prev_health = info.get("health", 9)
        self.unlocked_this_episode = set()
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_belief_reward = 0.0

        # Return (obs, info) tuple for Gymnasium/modern SB3 compatibility
        return obs, info

    def _compute_belief_reward(self, achievement, env_reward):
        n = self.achievement_counts[achievement]
        prior = self.prior_means.get(achievement, 0.0)
        if n == 0:
            belief = prior
        else:
            weight_prior = self.lambda_param / (self.lambda_param + n)
            weight_env = n / (self.lambda_param + n)
            belief = weight_prior * prior + weight_env * env_reward
        self.achievement_counts[achievement] += 1
        self.current_belief_means[achievement] = belief
        return belief

    def _compute_cluster_belief_reward(self, achievement, env_reward):
        cluster_name = None
        for cluster, achievements in self.belief_clusters.items():
            if achievement in achievements:
                cluster_name = cluster
                break
        if cluster_name is None:
            return 0.0
        n = self.cluster_counts[cluster_name]
        cluster_priors = [self.prior_means.get(a, 0.0) for a in self.belief_clusters[cluster_name]]
        prior = np.mean(cluster_priors) if cluster_priors else 0.0
        if n == 0:
            cluster_belief = prior
        else:
            weight_prior = self.lambda_param / (self.lambda_param + n)
            weight_env = n / (self.lambda_param + n)
            cluster_belief = weight_prior * prior + weight_env * env_reward
        self.cluster_counts[cluster_name] += 1
        self.cluster_belief_means[cluster_name] = cluster_belief
        return cluster_belief

    def step(self, action):
        """
        Gymnasium-compatible step() that returns 5-tuple.
        Returns: (obs, shaped_reward, terminated, truncated, info)
        """
        result = self.env.step(action)

        # Unpack robustly for Gymnasium 5-tuple
        if len(result) == 5:
            obs, env_reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, env_reward, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step() output: {result}")

        info['original_reward'] = env_reward
        self.episode_original_reward += env_reward

        # Health shaping
        health = info.get("health", self.prev_health)
        health_delta = health - self.prev_health
        health_reward = self.health_weight * health_delta

        # Belief rewards
        achievements = info.get("achievements", {})
        belief_total = 0.0
        new_achievements = 0

        for ach, unlocked in achievements.items():
            if unlocked and ach not in self.unlocked_this_episode:
                self.unlocked_this_episode.add(ach)
                self.unlocked_ever.add(ach)
                new_achievements += 1
                env_ach_reward = 1.0
                individual_belief = self._compute_belief_reward(ach, env_ach_reward)
                if self.use_clusters:
                    cluster_belief = self._compute_cluster_belief_reward(ach, env_ach_reward)
                    belief_reward = 0.5 * individual_belief + 0.5 * cluster_belief
                else:
                    belief_reward = individual_belief
                belief_total += belief_reward

        if self.clip_belief_reward:
            belief_total = np.clip(belief_total, -5.0, 5.0)

        # Total shaped reward
        shaped_reward = (env_reward - new_achievements + belief_total + health_reward)

        info['belief_reward'] = belief_total
        info['health_reward'] = health_reward
        info['shaped_reward'] = shaped_reward
        info['num_new_achievements'] = new_achievements

        self.episode_belief_reward += belief_total
        self.episode_shaped_reward += shaped_reward
        self.prev_health = health

        if terminated or truncated:
            info['episode_original_reward'] = self.episode_original_reward
            info['episode_shaped_reward'] = self.episode_shaped_reward
            info['episode_belief_reward'] = self.episode_belief_reward
            info['episode_achievements_unlocked'] = len(self.unlocked_this_episode)

        # Return Gymnasium 5-tuple format
        return obs, shaped_reward, terminated, truncated, info

    def render(self, mode='rgb_array', **kwargs):
        """Pass through render calls to the wrapped environment"""
        try:
            if hasattr(self.env, 'render'):
                return self.env.render(mode=mode, **kwargs)
            elif hasattr(self.unwrapped, 'render'):
                return self.unwrapped.render(mode=mode, **kwargs)
        except Exception as e:
            # Silently return None if rendering fails
            return None

    def get_belief_stats(self):
        return {
            'achievement_counts': dict(self.achievement_counts),
            'current_belief_means': dict(self.current_belief_means),
            'cluster_counts': dict(self.cluster_counts),
            'cluster_belief_means': dict(self.cluster_belief_means),
            'unlocked_this_episode': list(self.unlocked_this_episode),
            'total_unlocked_ever': len(self.unlocked_ever),
            'unlocked_ever': list(self.unlocked_ever)
        }

    def get_average_belief_by_tier(self):
        tiers = {
            'Tier 1 (Basic)': ['collect_drink', 'collect_wood', 'collect_sapling', 'wake_up'],
            'Tier 2 (Early)': ['place_table', 'eat_cow', 'place_plant', 'defeat_zombie'],
            'Tier 3 (Tools)': ['make_wood_pickaxe', 'make_wood_sword', 'collect_stone'],
            'Tier 4 (Advanced)': ['make_stone_pickaxe', 'make_stone_sword', 'collect_coal',
                                  'place_stone', 'defeat_skeleton'],
            'Tier 5 (Furnace)': ['place_furnace', 'collect_iron'],
            'Tier 6 (Iron)': ['make_iron_pickaxe', 'make_iron_sword'],
            'Tier 7 (Hardest)': ['collect_diamond', 'eat_plant']
        }
        tier_stats = {}
        for tier_name, achievements in tiers.items():
            beliefs = [self.current_belief_means[a] for a in achievements
                       if a in self.current_belief_means]
            tier_stats[tier_name] = np.mean(beliefs) if beliefs else 0.0
        return tier_stats