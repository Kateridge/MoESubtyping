"""Train the MoE method without evaluation, plotting, or benchmark code."""

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import prepare_training_data
from losses import classification_loss, guidance_factor, router_regularization, survival_loss
from model import SubtypingMoE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared training CSV.")
    parser.add_argument("--task", choices=["classification", "survival", "simulation"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="New/empty output directory.")
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64, help="0 uses the full training set.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dropout", type=float, default=None)
    parser.add_argument("--router-dropout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0, help="Gumbel-softmax temperature.")
    parser.add_argument("--initializer", choices=["kmeans", "gmm"], default="kmeans")
    parser.add_argument("--pseudo-labels", type=Path, help="Optional externally generated guidance labels.")
    parser.add_argument("--guidance-epochs", type=int, default=20, help="Epochs until guidance reaches zero.")
    parser.add_argument("--guidance-power", type=float, default=1.0, help="Decay curve exponent; 1 is linear.")
    parser.add_argument("--loss-guide", type=float, default=None, help="Initial guidance weight.")
    parser.add_argument("--loss-balance", type=float, default=None)
    parser.add_argument("--loss-sparsity", type=float, default=None)
    parser.add_argument("--add-covariates", action="store_true", help="Append age/sex; disabled by default.")
    parser.add_argument("--covariates", type=Path, help="Optional subject_id,age,sex CSV.")
    args = parser.parse_args(argv)
    simulated = args.task == "simulation"
    defaults = {
        "experts": 3 if simulated else 4, "epochs": 50 if simulated else 80,
        "lr": 0.005 if simulated else 0.02,
        "expert_dropout": 0.2 if simulated else 0.5,
        "router_dropout": 0.2 if args.task == "classification" else 0.0,
        "loss_guide": 0.1 if simulated else 0.2,
        "loss_balance": 0.4 if simulated else 0.6,
        "loss_sparsity": 0.25 if simulated else 0.2,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("lr", "temperature", "guidance_power"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("loss_guide", "loss_balance", "loss_sparsity"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.experts < 2 or args.epochs < 1 or args.hidden_dim < 1:
        parser.error("Require experts >= 2, epochs >= 1, hidden-dim >= 1")
    if args.batch_size < 0 or args.guidance_epochs < 0 or not 0 <= args.seed < 2**32:
        parser.error("Require nonnegative batch-size/guidance-epochs and seed in [0, 2**32)")
    if not 0 <= args.expert_dropout < 1 or not 0 <= args.router_dropout < 1:
        parser.error("Dropout probabilities must be in [0, 1)")
    if args.covariates is not None and not args.add_covariates:
        parser.error("--covariates requires --add-covariates")
    return args


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Output directory must be new or empty to avoid overwriting a run.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, preprocessing = prepare_training_data(
        args.data, args.task, args.experts, args.seed, args.initializer,
        args.pseudo_labels, args.add_covariates, args.covariates,
    )
    model_config = dict(
        input_dim=len(preprocessing["features"]), experts=args.experts,
        hidden_dim=args.hidden_dim, expert_dropout=args.expert_dropout,
        router_dropout=args.router_dropout,
    )
    model = SubtypingMoE(**model_config).to(args.device)
    # One optimizer is equivalent to the original two SGD optimizers because
    # all parameters use the same learning rate/momentum and update together.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loader = DataLoader(
        dataset, batch_size=args.batch_size or len(dataset), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    fields = ["epoch", "total", "task", "balance", "sparsity", "guidance", "guidance_weight"]
    with (args.output / "training_loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            totals = dict.fromkeys(fields[1:6], 0.0)
            guide_weight = args.loss_guide * guidance_factor(
                epoch, args.guidance_epochs, args.guidance_power
            )
            for batch in loader:
                x, target, event, time, pseudo = [value.to(args.device) for value in batch]
                optimizer.zero_grad(set_to_none=True)
                logits, outputs = model(x)
                assignments = F.gumbel_softmax(logits, tau=args.temperature, hard=True, dim=1)
                if args.task == "survival":
                    task_loss = survival_loss(outputs, event, time, assignments)
                else:
                    if args.task == "simulation":
                        # Controls train every expert; only simulated patients
                        # receive hard subtype assignments, as in the source.
                        assignments = torch.where(
                            (target == 1)[:, None], assignments,
                            torch.full_like(assignments, 1.0 / args.experts),
                        )
                    task_loss = classification_loss(outputs, target, assignments)
                balance, sparsity, guidance = router_regularization(logits, pseudo)
                # Task-only ablation: total = task_loss
                total = (task_loss + args.loss_balance * balance
                         + args.loss_sparsity * sparsity + guide_weight * guidance)
                if not torch.isfinite(total):
                    raise FloatingPointError("Nonfinite training loss; inspect data and learning rate.")
                total.backward()
                optimizer.step()
                for name, value in zip(totals, (total, task_loss, balance, sparsity, guidance)):
                    totals[name] += value.detach().item()
            row = {name: value / len(loader) for name, value in totals.items()}
            row.update(epoch=epoch + 1, guidance_weight=guide_weight)
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={row['total']:.6f} "
                  f"task={row['task']:.6f} guidance_weight={guide_weight:.4f}")

    torch.save({
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": model_config, "preprocessing": preprocessing,
        "training_config": config, "epoch": args.epochs,
    }, args.output / "model.pt")
    print(f"Saved trained model to {args.output / 'model.pt'}")
    return model


if __name__ == "__main__":
    train(parse_args())
