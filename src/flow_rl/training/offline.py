"""The same minibatch loop supports supervised and offline RL algorithms."""

from flow_rl.training.build import to_torch


def train_offline(algorithm, dataset, config, device, start_step, on_step):
    for step in range(start_step + 1, config.training.steps + 1):
        batch = to_torch(dataset.sample(config.training.batch_size), device)
        on_step(step, algorithm.update(batch))
    return max(start_step, config.training.steps)
