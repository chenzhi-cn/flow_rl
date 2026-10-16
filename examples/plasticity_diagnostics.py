"""Run five diagnostics on the existing MLP and diffusion policies, without a simulator."""

import argparse
import json
from pathlib import Path

import torch

from flow_rl.diagnostics import PlasticityMonitor
from flow_rl.models import DiffusionChunkPolicy, GaussianChunkPolicy


def measure(name: str, num_samples: int, batch_size: int) -> dict:
    torch.manual_seed(7)
    if name == "mlp":
        model = GaussianChunkPolicy(28, 5, chunk_size=4, hidden_dims=(64, 64))
        layer = "net.3"  # Second GELU output: [B, 64].
    else:
        model = DiffusionChunkPolicy(28, 5, chunk_size=4, hidden_dim=32, num_layers=1, num_heads=4)
        layer = "blocks.0.ffn.1"  # SiLU output: [B, 5 tokens, 64].
    observations = torch.randn(num_samples, 28)
    actions = torch.rand(num_samples, 4, 5) * 1.8 - 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # A single registration observes the chosen layer in training and evaluation.
    with PlasticityMonitor(model, layer=layer) as probe:
        model.train()
        for obs, act in zip(observations.split(batch_size), actions.split(batch_size)):
            optimizer.zero_grad()
            model.per_sample_loss(obs, act).mean().backward()
            optimizer.step()
        training = probe.summary()  # Last batch, weights captured during forward.

        model.eval()
        probe.reset()  # Start a new dataset (also needed for repeated eval passes).
        with torch.no_grad():
            for obs, act in zip(observations.split(batch_size), actions.split(batch_size)):
                if name == "diffusion":
                    # Fixed probe noise/time makes this representation reproducible.
                    model.per_sample_loss(
                        obs, act, noise=torch.zeros_like(act), time=torch.full((len(obs),), 0.5)
                    )
                else:
                    model.per_sample_loss(obs, act)
        evaluation = probe.summary()
    return {"training_last_batch": training, "evaluation": evaluation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "diffusion", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=129)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("num-samples and batch-size must be positive")
    torch.set_num_threads(1)
    names = ("mlp", "diffusion") if args.model == "both" else (args.model,)
    result = {
        "data": "synthetic state/action samples; functional diagnostics demonstration",
        "seed": 7,
        "torch_version": torch.__version__,
        "input_samples": args.num_samples,
        "batch_size": args.batch_size,
        "models": {name: measure(name, args.num_samples, args.batch_size) for name in names},
    }
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    main()
