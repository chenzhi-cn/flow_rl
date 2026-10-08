"""Command-line entrypoints; all implementation lives in reusable modules."""

import argparse
import json

from flow_rl.config import load_config


def parser():
    root = argparse.ArgumentParser(description="Modular robotics training and chunk inference")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "check-config", "collect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=name != "evaluate")
        cmd.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
        )
        if name == "train":
            group = cmd.add_mutually_exclusive_group()
            group.add_argument(
                "--resume", help="Continue an offline run with optimizer and RNG state"
            )
            group.add_argument(
                "--init-from",
                help="Load model/optimizer state for a new run, e.g. online fine-tuning",
            )
        elif name == "evaluate":
            cmd.add_argument("--checkpoint", required=True)
        elif name == "collect":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--steps", type=int, default=400)
    return root


def main():
    args = parser().parse_args()
    if args.command == "evaluate":
        from flow_rl.config import apply_overrides, from_dict
        from flow_rl.evaluation import evaluate
        from flow_rl.training.build import make_env, setup
        from flow_rl.training.checkpoints import load_policy

        policy, saved = load_policy(args.checkpoint)
        config = load_config(args.config, args.overrides) if args.config else saved
        if not args.config and args.overrides:
            config = from_dict(apply_overrides(saved.to_dict(), args.overrides))
        if config.environment != saved.environment or config.model != saved.model:
            raise ValueError(
                "Evaluation must use the checkpoint's environment and model configuration"
            )
        policy.to(setup(config))
        env = make_env(config)
        try:
            result = evaluate(policy, env, config)
        finally:
            env.close()
    else:
        config = load_config(args.config, args.overrides)
        if args.command == "check-config":
            result = config.to_dict()
        elif args.command == "collect":
            from flow_rl.data import collect_random_dataset, save_npz
            from flow_rl.training.build import make_env, setup

            setup(config)
            env = make_env(config)
            try:
                data = collect_random_dataset(env, args.steps, config.experiment.seed)
                save_npz(args.output, data)
            finally:
                env.close()
            result = {"output": args.output, "transitions": args.steps, "source": "random"}
        else:
            from flow_rl.training.engine import train

            result = train(config, args.resume, args.init_from)
    print(json.dumps(result, indent=2, allow_nan=False))
