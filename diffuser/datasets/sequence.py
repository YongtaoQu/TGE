from collections import namedtuple
import collections
import numpy as np
import torch
import pdb
import pickle

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env, horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=50000, termination_penalty=0, use_padding=True, seed=None,
        data_path=None, dataset_dict=None):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        if dataset_dict is not None:
             # Load from dictionary directly
             self._load_from_dict(dataset_dict, normalizer, max_n_episodes, termination_penalty, max_path_length, seed, env)
        elif data_path is not None:
            # Load from pickle file
            self._load_from_pickle(data_path, normalizer, max_n_episodes, termination_penalty, max_path_length, seed, env)
        else:
            # Standard loading from environment
            print(f"[ SequenceDataset ] Loading from environment: {env}")
            self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
            self.env = env = load_environment(env)
            self.env.seed(seed)
            itr = sequence_dataset(env, self.preprocess_fn)
            fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
            for i, episode in enumerate(itr):
                fields.add_path(episode)
            fields.finalize()
            self._setup_dataset(fields, normalizer)

        print(self.fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')
    
    def _load_from_dict(self, dataset_dict, normalizer, max_n_episodes, termination_penalty, max_path_length, seed, env=None):
        """Load dataset from provided dictionary."""
        print(f"[ SequenceDataset ] Loading dataset from provided dictionary")
        
        data = dataset_dict
        
        # Extract data components
        observations = data['observations']
        actions = data['actions'] 
        rewards = data['rewards']
        terminals = data['terminals']
        next_observations = data['next_observations']
        
        print(f"[ SequenceDataset ] Loaded {len(observations):,} transitions")
        print(f"[ SequenceDataset ] Observation dim: {observations.shape[1]}")
        print(f"[ SequenceDataset ] Action dim: {actions.shape[1]}")
        
        # Infer env name
        if env is not None:
             # Same inference logic as _load_from_pickle
             env_lower = str(env).lower()
             if 'mixed_' in env_lower or not '-' in str(env):
                if 'walker2d' in env_lower or 'walker' in env_lower:
                    env_name = 'walker2d-random-v2'
                elif 'hopper' in env_lower:
                    env_name = 'hopper-random-v2'
                elif 'halfcheetah' in env_lower:
                    env_name = 'halfcheetah-random-v2'
                elif 'antmaze' in env_lower:
                    env_name = 'antmaze-umaze-v0'
                elif 'ant' in env_lower:
                    env_name = 'ant-random-v2'
                elif 'door' in env_lower:
                    env_name = 'door-human-v1'
                elif 'pen' in env_lower:
                     env_name = 'pen-human-v1'
                elif 'hammer' in env_lower:
                    env_name = 'hammer-human-v1'
                elif 'relocate' in env_lower:
                    env_name = 'relocate-human-v1'
                elif 'kitchen' in env_lower:
                    env_name = 'kitchen-mixed-v0'
                else:
                    env_name = None
             else:
                 env_name = env
        else:
             env_name = None
             
        if env_name is None:
            raise ValueError("Environment name must be provided when loading from dictionary")

        self.env = load_environment(env_name)
        if seed is not None:
            self.env.seed(seed)
            
        # Create ReplayBuffer
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        # MockEnv class
        class MockEnv:
            def __init__(self, observations, actions, rewards, terminals, next_observations, max_path_length):
                self.observations = observations
                self.actions = actions
                self.rewards = rewards
                self.terminals = terminals
                self.next_observations = next_observations
                self._max_episode_steps = max_path_length
                self.name = 'mixed_dataset'
                
                # Only set timeouts if present in data, otherwise d4rl.sequence_dataset
                # will fall back to max_episode_steps logic which is safer if timeouts are missing
                if 'timeouts' in data:
                    self.timeouts = data['timeouts'].copy()
                elif 'dones' in data:
                    self.timeouts = data['dones'].copy().astype(bool)
                else:
                    self.timeouts = None
            
            def get_dataset(self):
                dataset = {
                    'observations': self.observations,
                    'actions': self.actions,
                    'rewards': self.rewards,
                    'terminals': self.terminals,
                    'next_observations': self.next_observations,
                }
                if self.timeouts is not None:
                    dataset['timeouts'] = self.timeouts
                return dataset
        
        mock_env = MockEnv(observations, actions, rewards, terminals, next_observations, max_path_length)
        
        
        itr = sequence_dataset(mock_env, lambda x: x)
        episode_count = 0
        for episode in itr:
            episode_length = len(episode['observations'])
            if episode_length > max_path_length:
                print(f"[ SequenceDataset ] Warning: Episode {episode_count} has length {episode_length} > {max_path_length}, truncating to {max_path_length}")
                truncated_episode = {}
                for key in episode:
                    if isinstance(episode[key], np.ndarray) and len(episode[key]) > max_path_length:
                        truncated_episode[key] = episode[key][:max_path_length]
                    else:
                        truncated_episode[key] = episode[key]
                if 'terminals' in truncated_episode:
                    truncated_episode['terminals'][-1] = True
                episode = truncated_episode
            
            fields.add_path(episode)
            episode_count += 1
            
        print(f"[ SequenceDataset ] Loaded {episode_count:,} trajectories (episodes)")
        
        # Ensure keys attribute is set on fields if it wasn't set (e.g. if episode_count is 0)
        if not hasattr(fields, 'keys'):
             fields.keys = ['observations', 'actions', 'rewards', 'terminals', 'next_observations', 'timeouts']
             
        fields.finalize()
        
        self._setup_dataset(fields, normalizer)

    def _load_from_pickle(self, data_path, normalizer, max_n_episodes, termination_penalty, max_path_length, seed, env=None):
        """Load dataset from pickle file containing mixed data."""
        print(f"[ SequenceDataset ] Loading mixed dataset from: {data_path}")
        
        # Load the mixed data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data components
        observations = data['observations']
        actions = data['actions'] 
        rewards = data['rewards']
        terminals = data['terminals']
        next_observations = data['next_observations']
        
        print(f"[ SequenceDataset ] Loaded {len(observations):,} transitions")
        print(f"[ SequenceDataset ] Observation dim: {observations.shape[1]}")
        print(f"[ SequenceDataset ] Action dim: {actions.shape[1]}")
        
        # Set up environment info - use provided env or infer from data_path
        # First, try to infer from env parameter (might be a mixed dataset name)
        if env is not None:
            env_lower = str(env).lower()
            # Check if env is a mixed dataset name (contains 'mixed_') or needs inference
            if 'mixed_' in env_lower or not '-' in str(env):
                # Infer from env name
                if 'walker2d' in env_lower or 'walker' in env_lower:
                    env_name = 'walker2d-random-v2'
                elif 'hopper' in env_lower:
                    env_name = 'hopper-random-v2'
                elif 'halfcheetah' in env_lower:
                    env_name = 'halfcheetah-random-v2'
                elif 'ant' in env_lower:
                    env_name = 'ant-random-v2'
                elif 'door' in env_lower:
                    env_name = 'door-human-v1'
                elif 'pen' in env_lower:
                    env_name = 'pen-human-v1'
                elif 'hammer' in env_lower:
                    env_name = 'hammer-human-v1'
                elif 'kitchen' in env_lower:
                    env_name = 'kitchen-mixed-v0'
                else:
                    # If we can't infer from env, try data_path
                    env_name = None
            else:
                # env is already a valid gym environment name
                env_name = env
        
        # If we couldn't infer from env, try data_path
        if env_name is None:
            # Extract environment name from data_path as fallback
            if 'walker2d' in data_path.lower() or 'walker' in data_path.lower():
                env_name = 'walker2d-random-v2'
            elif 'hopper' in data_path.lower():
                env_name = 'hopper-random-v2'
            elif 'halfcheetah' in data_path.lower():
                env_name = 'halfcheetah-random-v2'
            elif 'antmaze' in data_path.lower():
                env_name = 'antmaze-umaze-v0'
            elif 'ant' in data_path.lower():
                env_name = 'ant-random-v2'
            elif 'door' in data_path.lower():
                env_name = 'door-human-v1'
            elif 'pen' in data_path.lower():
                env_name = 'pen-human-v1'
            elif 'hammer' in data_path.lower():
                env_name = 'hammer-human-v1'
            elif 'kitchen' in data_path.lower():
                env_name = 'kitchen-mixed-v0'
            else:
                raise ValueError(f"Unsupported environment. Could not infer from env={env} or data_path={data_path}")
        
        self.env = load_environment(env_name)
        if seed is not None:
            self.env.seed(seed)
        
        # Create ReplayBuffer like the original SequenceDataset
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        # Create a mock environment to use with sequence_dataset
        class MockEnv:
            def __init__(self, observations, actions, rewards, terminals, next_observations, max_path_length):
                self.observations = observations
                self.actions = actions
                self.rewards = rewards
                self.terminals = terminals
                self.next_observations = next_observations
                self._max_episode_steps = max_path_length
                self.name = 'mixed_dataset'
                
                # Use original terminals and timeouts from data, don't force max_path_length splits
                # Check if timeouts field exists in the data
                if 'timeouts' in data:
                    self.terminals = terminals.copy()
                    self.timeouts = data['timeouts'].copy()
                else:
                    # Fallback: use terminals as episode boundaries
                    self.terminals = terminals.copy()
                    self.timeouts = np.zeros_like(terminals)
            
            def get_dataset(self):
                return {
                    'observations': self.observations,
                    'actions': self.actions,
                    'rewards': self.rewards,
                    'terminals': self.terminals,  # Use modified terminals
                    'next_observations': self.next_observations,
                    'timeouts': self.timeouts,
                }
        
        # Create mock environment and use sequence_dataset
        mock_env = MockEnv(observations, actions, rewards, terminals, next_observations, max_path_length)
        
        # Use the original sequence_dataset function
        itr = sequence_dataset(mock_env, lambda x: x)  # No preprocessing needed
        episode_count = 0
        skipped_count = 0
        for episode in itr:
            # Check if episode length exceeds max_path_length and truncate if necessary
            episode_length = len(episode['observations'])
            if episode_length > max_path_length:
                print(f"[ SequenceDataset ] Warning: Episode {episode_count} has length {episode_length} > {max_path_length}, truncating to {max_path_length}")
                # Truncate all fields to max_path_length
                truncated_episode = {}
                for key in episode:
                    if isinstance(episode[key], np.ndarray) and len(episode[key]) > max_path_length:
                        truncated_episode[key] = episode[key][:max_path_length]
                    else:
                        truncated_episode[key] = episode[key]
                # Mark the last step as terminal to indicate truncation
                if 'terminals' in truncated_episode:
                    truncated_episode['terminals'][-1] = True
                episode = truncated_episode
            
            fields.add_path(episode)
            episode_count += 1
        
        print(f"[ SequenceDataset ] Loaded {episode_count:,} trajectories (episodes)")
        fields.finalize()
        
        self._setup_dataset(fields, normalizer)
    
    def _setup_dataset(self, fields, normalizer):
        """Setup dataset fields, normalizer, and indices."""
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, self.horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

class ImitationValueDataset(ValueDataset):
    def __init__(self, *args, q_function=None, device='cuda', discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_function = q_function.to(device)
        self.device = device
        self.discount = discount

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        trajectory = batch.trajectories  # [H, obs+act]
        obs = trajectory[:, :self.observation_dim]
        act = trajectory[:, self.observation_dim:]

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.q_function(obs, act).squeeze(-1)  # [H]
            discounts = self.discount ** torch.arange(len(q_values), device=self.device)
            value = torch.sum(q_values * discounts).cpu().numpy()

        value = np.array([value], dtype=np.float32)
        return ValueBatch(*batch[:-1], value)
