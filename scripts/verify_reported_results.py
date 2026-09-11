"""Rerun saved policies with the original evaluator; never train or query an LLM.

Run with the existing IPR conda interpreter. Outputs are separate from source runs.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/Users/zfy/Projects/meta/llm-log-analysis')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(model, weights):
    return {
        'source_model': str(model), 'model_sha256': sha(model),
        'source_checkpoint': str(weights), 'checkpoint_sha256': sha(weights),
        'evaluator': str(SOURCE / 'utils.py'),
        'evaluator_sha256': sha(SOURCE / 'utils.py'),
        'python': sys.executable,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch', 'numpy', 'gymnasium', 'metaworld', 'mujoco', 'box2d-py']},
        'kind': 'fresh inference from archived checkpoint, not an archived performance log',
        'device': 'cpu', 'torch_threads': 1,
    }


def evaluator():
    import torch
    import utils
    torch.set_num_threads(1)
    utils.device = 'cpu'
    utils.tqdm = functools.partial(utils.tqdm, disable=True)
    return torch, utils


def car_worker(job):
    model, weights, seed, sample, max_steps = job
    torch, utils = evaluator()
    output = io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(output), torch.inference_mode():
        frame, total = utils.evaluate_policy_single(
            model, weights, env_name='racecar', max_steps=max_steps,
            save_video=False, sample=sample, log_data=True, seed=seed,
            torch_seed=0, num_episodes=1, include_reward=True,
        )
    return {'seed': seed, 'torch_seed': 0, 'return': float(total),
            'steps': len(frame), 'seconds': time.monotonic() - start,
            'stdout': output.getvalue()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['racecar', 'door'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    model, weights = Path(args.model).resolve(), Path(args.weights).resolve()
    if not model.is_file() or not weights.is_file():
        raise FileNotFoundError('Both source model and checkpoint are required.')
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing verification: {destination}')
    result = metadata(model, weights)
    result['settings'] = vars(args)
    result['settings']['seed'] = args.seed if args.seed is not None else (1000 if args.task == 'racecar' else 0)
    result['settings']['max_steps'] = args.max_steps or (1000 if args.task == 'racecar' else 500)
    start = time.monotonic()
    print(f'Start {args.task}: {model}; checkpoint {weights}; sample={args.sample}', flush=True)
    import numpy as np
    if args.task == 'racecar':
        jobs = [(str(model), str(weights), seed, args.sample, result['settings']['max_steps'])
                for seed in range(result['settings']['seed'], result['settings']['seed'] + args.episodes)]
        result['episodes'] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(car_worker, job) for job in jobs]
            for future in as_completed(futures):
                episode = future.result()
                result['episodes'].append(episode)
                print(f"Seed {episode['seed']}: {episode['return']:.8f}, {episode['steps']} steps", flush=True)
        result['episodes'].sort(key=lambda x: x['seed'])
        returns = np.array([x['return'] for x in result['episodes']])
        result['summary'] = {'mean': float(returns.mean()), 'std_population': float(returns.std())}
    else:
        torch, utils = evaluator()
        original_make = utils.gym.make
        episodes = []

        class ObserveDistance(utils.gym.Wrapper):
            """Record evaluation quantities without changing observations/actions."""
            def reset(self, **kwargs):
                observation, info = self.env.reset(**kwargs)
                base = self.unwrapped
                episodes.append({'seed': kwargs.get('seed'), 'initial_handle': base.data.geom('handle').xpos.copy().tolist(),
                                 'target': base._target_pos.copy().tolist(),
                                 'initial_observation': observation.copy().tolist(), 'distances': []})
                return observation, info

            def step(self, action):
                transition = self.env.step(action)
                base = self.unwrapped
                episodes[-1]['distances'].append(float(np.linalg.norm(base.data.geom('handle').xpos - base._target_pos)))
                return transition

        utils.gym.make = lambda *a, **kw: ObserveDistance(original_make(*a, **kw))
        try:
            with torch.inference_mode():
                original_result = utils.evaluate_policy_door(
                    str(model), str(weights), max_steps=result['settings']['max_steps'],
                    sample=args.sample, save_video=False, seed=result['settings']['seed'],
                    torch_seed=0, num_episodes=args.episodes,
                )
        finally:
            utils.gym.make = original_make
        result['original_evaluator_result'] = {k: float(v) for k, v in original_result.items()}
        result['episodes'] = episodes
        for index, episode in enumerate(episodes):
            episode.update(torch_seed=index, steps=len(episode['distances']),
                           min_distance=min(episode['distances']), final_distance=episode['distances'][-1])
            print(f"Seed {episode['seed']}: min {episode['min_distance']:.9f}; final {episode['final_distance']:.9f}", flush=True)
        mins = np.array([x['min_distance'] for x in episodes])
        ends = np.array([x['final_distance'] for x in episodes])
        result['summary'] = {'mean_min_distance': float(mins.mean()), 'std_min_distance_population': float(mins.std()),
                             'mean_final_distance': float(ends.mean()), 'std_final_distance_population': float(ends.std())}
        assert np.isclose(mins.mean(), original_result['min_distance'])
        assert np.isclose(ends.mean(), original_result['mean_end_distance'])
        assert np.isclose(ends.std(), original_result['std_end_distance'])
    result['elapsed_seconds'] = time.monotonic() - start
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']), flush=True)
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()
