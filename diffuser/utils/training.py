import os
import copy
import numpy as np
import torch
import einops
import pdb
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        eval_freq=1000,
        n_eval_episodes=5,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = next(diffusion_model.parameters()).device
        self.device = device
        
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                
                import wandb
                log_dict = {
                    'train/loss': loss.item(),
                    'train/step': self.step,
                    'train/time': timer(),
                }
                for key, val in infos.items():
                    log_dict[f'train/{key}'] = val
                wandb.log(log_dict)

            if self.eval_freq and self.step > 0 and self.step % self.eval_freq == 0:
                eval_results = self.evaluate(n_episodes=self.n_eval_episodes, verbose=True)
                
                import wandb
                wandb.log({
                    'eval/mean_score': eval_results['eval/mean_score'],
                    'eval/std_score': eval_results['eval/std_score'],
                    'eval/step': self.step,
                })

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def evaluate(self, n_episodes=5, max_episode_length=None, verbose=False):
        '''
            evaluates the current ema model without value guide
        '''
        from diffuser.sampling.policies import GuidedPolicy
        
        policy = GuidedPolicy(
            guide=None,  # No value guide
            diffusion_model=self.ema_model,
            normalizer=self.dataset.normalizer,
            preprocess_fns=[],
        )
        
        env = self.dataset.env
        
        if max_episode_length is None:
            if hasattr(env, '_max_episode_steps'):
                 max_episode_length = env._max_episode_steps
            else:
                env_name = env.name if hasattr(env, 'name') else str(env)
                if 'pen' in env_name.lower():
                    max_episode_length = 100
                elif 'door' in env_name.lower() or 'relocate' in env_name.lower() or 'hammer' in env_name.lower():
                    max_episode_length = 200
                elif 'kitchen' in env_name.lower():
                    max_episode_length = 280
                else:
                    max_episode_length = 1000
        scores = []
        for i in range(n_episodes):
            total_reward = 0
            try:
                observation = env.reset(seed=i)
            except TypeError:
                observation = env.reset()
            
            if verbose:
                pbar = tqdm(range(max_episode_length), desc=f'Eval Episode {i+1}')
            else:
                pbar = range(max_episode_length)
                
            for t in pbar:
                conditions = {0: observation}
                action, _ = policy(conditions, batch_size=1, verbose=False)
                
                next_observation, reward, terminal, _ = env.step(action)
                total_reward += reward
                
                if verbose:
                    score = env.get_normalized_score(total_reward) * 100.0
                    pbar.set_postfix({
                        't': t,
                        'r': f'{reward:.2f}',
                        'R': f'{total_reward:.2f}',
                        'score': f'{score:.4f}',
                    })
                
                if terminal:
                    break
                    
                observation = next_observation
            
            score = env.get_normalized_score(total_reward) * 100.0
            scores.append(score)
            
            if verbose:
                print(f"Eval Episode {i+1} final score: {score:.2f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if verbose:
            print(f"\nEval Mean score over {len(scores)} episodes: {mean_score:.2f}")
            print(f"Eval Std score over {len(scores)} episodes: {std_score:.2f}")
        
        return {
            'eval/mean_score': mean_score,
            'eval/std_score': std_score,
            'eval/scores': scores,
        }

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)

            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            normed_conditions = to_np(batch.conditions[0])[:,None]

            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
