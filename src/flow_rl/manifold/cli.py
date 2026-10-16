"""Two-stage command line workflow and external-prediction scoring."""

import argparse
import json
import sys

from flow_rl.manifold.artifact import load_bundle
from flow_rl.manifold.config import load_config


def _progress(message):
    print(message, file=sys.stderr, flush=True)


def parser():
    root = argparse.ArgumentParser(description="Frozen-sample manifold adherence evaluation")
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Generate once, or verify and reuse the existing samples"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    compare = sub.add_parser("compare", help="Compare checkpoints without a simulator")
    compare.add_argument("--config")
    compare.add_argument("--samples")
    compare.add_argument("--checkpoint", action="append", metavar="LABEL=PATH")
    compare.add_argument("--output-dir")
    compare.add_argument("--device")
    compare.add_argument("--batch-size", type=int)
    compare.add_argument("--seed", type=int)
    compare.add_argument("--stochastic", action="store_true", default=None)
    inspect = sub.add_parser("inspect", help="Verify checksum and summarize sample geometry")
    inspect.add_argument("--samples", required=True)
    score = sub.add_parser("score", help="Score externally generated [N*m,H,A] predictions")
    score.add_argument("--samples", required=True)
    score.add_argument(
        "--predictions", required=True, help=".npy array in exact stored sample order"
    )
    score.add_argument("--label", default="external")
    score.add_argument("--action-space", choices=["normalized", "physical"], default="physical")
    score.add_argument("--output-dir", required=True)
    return root


def _summary(bundle):
    ranks = bundle.arrays["ranks"]
    return {
        "sample_id": bundle.sample_id,
        "sample_count": bundle.metadata["sample_count"],
        "preparation": bundle.metadata["preparation"],
        "action_vector_dim": bundle.metadata["action_vector_dim"],
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "full_rank_fraction": float((ranks == bundle.metadata["action_vector_dim"]).mean()),
        "clipped_perturbation_count": bundle.metadata["clipped_perturbation_count"],
        "restoration_statistics": bundle.metadata["restoration_statistics"],
    }


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "prepare":
        from flow_rl.manifold.prepare import prepare_from_config

        config = load_config(args.config, args.set)
        bundle, reused = prepare_from_config(config, _progress)
        result = dict(_summary(bundle), samples=config["samples"], reused=reused)
    elif args.command == "inspect":
        result = _summary(load_bundle(args.samples))
    elif args.command == "score":
        import numpy as np

        from flow_rl.manifold.artifact import file_sha256
        from flow_rl.manifold.evaluate import evaluate_predictions, write_results

        bundle = load_bundle(args.samples)
        predictions = np.load(args.predictions, allow_pickle=False)
        summary, details = evaluate_predictions(predictions, bundle, action_space=args.action_space)
        summary.update(
            predictions_file=args.predictions, predictions_sha256=file_sha256(args.predictions)
        )
        report = write_results(args.output_dir, bundle, [(args.label, summary, details)])
        result = {
            "sample_id": bundle.sample_id,
            "output_dir": args.output_dir,
            "models": report["models"],
        }
    else:
        import torch

        from flow_rl.manifold.evaluate import compare_checkpoints

        config = load_config(args.config) if args.config else {}
        options = dict(config.get("comparison", {}))
        samples = args.samples or config.get("samples")
        if not samples:
            cli.error("compare requires --samples or --config")
        checkpoints = options.pop("checkpoints", {})
        if args.checkpoint:
            checkpoints = {}
            for entry in args.checkpoint:
                label, separator, path = entry.partition("=")
                if not separator or not label or not path or label in checkpoints:
                    cli.error("--checkpoint requires unique LABEL=PATH entries")
                checkpoints[label] = path
        output_dir = args.output_dir or options.pop("output_dir", None)
        options.pop("output_dir", None)
        if not output_dir:
            cli.error("compare requires --output-dir or comparison.output_dir")
        for key in ("device", "batch_size", "seed"):
            if getattr(args, key) is not None:
                options[key] = getattr(args, key)
        if args.stochastic:
            options["deterministic"] = False
        torch.set_num_threads(1)
        bundle = load_bundle(samples)
        report = compare_checkpoints(bundle, checkpoints, output_dir, progress=_progress, **options)
        result = {
            "sample_id": bundle.sample_id,
            "output_dir": output_dir,
            "models": report["models"],
        }
    print(json.dumps(result, indent=2, allow_nan=False))
