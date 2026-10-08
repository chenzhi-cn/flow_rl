"""Experiment lifecycle: build, train, checkpoint and evaluate."""

import json
import time
from pathlib import Path

from flow_rl.config import Config, save_config
from flow_rl.evaluation import evaluate
from flow_rl.training.build import build_algorithm, load_data, make_env, setup
from flow_rl.training.checkpoints import restore_training, save_checkpoint
from flow_rl.training.metrics import write_metrics
from flow_rl.training.offline import train_offline
from flow_rl.training.online import train_online


def train(config: Config, resume=None, init_from=None) -> dict:
    config.validate()
    device = setup(config)
    output = Path(config.experiment.output_dir)
    if (output / "checkpoint.pt").exists() and resume is None:
        raise FileExistsError(
            f"{output} already contains a run; set experiment.output_dir or use --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(config)
    evaluation_env = None
    try:
        print(f"Loading {config.data.source} data for {config.training.mode}", flush=True)
        dataset = load_data(config, env)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        algorithm = build_algorithm(config, observation_dim, action_dim, device, dataset)
        start_step = (
            restore_training(resume or init_from, config, algorithm, dataset, bool(resume))
            if resume or init_from
            else 0
        )
        save_config(config, output / "config.yaml")
        metadata = {
            "data_source": config.data.source,
            "dataset_transitions": dataset.size if dataset is not None else 0,
            "dataset_windows": len(dataset) if dataset is not None else 0,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "chunk_size": config.model.chunk_size,
            "actor_parameters": sum(p.numel() for p in algorithm.actor.parameters()),
            "device": str(device),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
        if config.training.eval_interval:
            evaluation_env = make_env(config)
        started = time.monotonic()

        def save(step, path=output / "checkpoint.pt"):
            save_checkpoint(path, config, algorithm, step, observation_dim, action_dim, dataset)

        def on_step(step, metrics):
            if (
                step == start_step + 1
                or step % config.training.log_interval == 0
                or step == config.training.steps
            ):
                write_metrics(
                    output / "metrics.jsonl",
                    {
                        "step": step,
                        "elapsed_seconds": time.monotonic() - started,
                        **metrics,
                    },
                )
            if (
                config.training.checkpoint_interval
                and step % config.training.checkpoint_interval == 0
            ):
                save(step)
            if evaluation_env is not None and step % config.training.eval_interval == 0:
                result = evaluate(algorithm.actor, evaluation_env, config)
                write_metrics(output / "evaluation.jsonl", {"step": step, **result})

        if config.training.mode == "online_rl":
            step = train_online(algorithm, dataset, env, config, device, on_step)
        else:
            step = train_offline(algorithm, dataset, config, device, start_step, on_step)
        save(step)
        result = evaluate(algorithm.actor, env, config)
        write_metrics(output / "evaluation.jsonl", {"step": step, **result})
        summary = {"checkpoint": str(output / "checkpoint.pt"), "step": step, **metadata, **result}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()
        if evaluation_env is not None:
            evaluation_env.close()
