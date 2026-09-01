from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .critic import ValueMLP, binomial_nll, fit_feature_pipeline
from .io import load_yaml, read_jsonl, stable_seed, write_json


def joined_rows(config, representation):
    prefixes = {row["state_id"]: row for row in read_jsonl(config["output"]["prefixes"])}
    problem_features = {}
    if representation == "budget-only":
        problem_features = {row["problem_id"]: row for row in read_jsonl(
            config["output"]["problem_features"]
        )}
    values = read_jsonl(config["output"]["state_values"])
    rows = []
    for value in values:
        state_id = f"{value['problem_id']}-b{value['spent_budget']}-m{value['prefix_id']}"
        row = {**value, **prefixes[state_id]}
        if representation == "budget-only":
            row.update(problem_features[value["problem_id"]])
        rows.append(row)
    return rows


def suite_split(rows, config):
    suites = sorted({row["suite_id"] for row in rows})
    rng = np.random.default_rng(config["training"]["bootstrap_seed"])
    rng.shuffle(suites)
    n_train = max(1, round(len(suites) * config["training"]["train_suite_fraction"]))
    n_val = max(1, round(len(suites) * config["training"]["validation_suite_fraction"]))
    train = set(suites[:n_train]); validation = set(suites[n_train:n_train+n_val])
    test = set(suites) - train - validation
    return train, validation, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--representation", choices=["state-aware", "budget-only"],
                        default="state-aware")
    args = parser.parse_args()
    config = load_yaml(args.config); training = config["training"]
    rows = joined_rows(config, args.representation)
    train_suites, val_suites, test_suites = suite_split(rows, config)
    train_rows = [row for row in rows if row["suite_id"] in train_suites]
    val_rows = [row for row in rows if row["suite_id"] in val_suites]
    state_aware = args.representation == "state-aware"
    pipeline = fit_feature_pipeline(
        train_rows, int(training["pca_components"]),
        hidden_key="last_hidden" if state_aware else "problem_hidden",
        state_aware=state_aware,
    )
    x_train = torch.tensor(pipeline.transform(train_rows)); x_val = torch.tensor(pipeline.transform(val_rows))
    y_train = torch.tensor([row["successes"] for row in train_rows], dtype=torch.float32)
    n_train = torch.tensor([row["trials"] for row in train_rows], dtype=torch.float32)
    y_val = torch.tensor([row["successes"] for row in val_rows], dtype=torch.float32)
    n_val = torch.tensor([row["trials"] for row in val_rows], dtype=torch.float32)
    output = Path(config["output"]["critic_dir"]) / args.representation
    output.mkdir(parents=True, exist_ok=True)
    pipeline.save(output)
    metrics = []
    for head in range(int(training["ensemble_heads"])):
        torch.manual_seed(stable_seed(training["bootstrap_seed"], head))
        model = ValueMLP(x_train.shape[1], int(training["hidden_size"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                      weight_decay=training["weight_decay"])
        rng = np.random.default_rng(stable_seed(training["bootstrap_seed"], "bootstrap", head))
        indices = torch.tensor(rng.integers(0, len(train_rows), len(train_rows)))
        best = float("inf"); best_state = None; stale = 0
        for epoch in range(int(training["epochs"])):
            model.train(); optimizer.zero_grad()
            loss = binomial_nll(model(x_train[indices]), y_train[indices], n_train[indices])
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad(): val = float(binomial_nll(model(x_val), y_val, n_val))
            if val < best - 1e-6:
                best = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; stale = 0
            else: stale += 1
            if stale >= int(training["patience"]): break
        model.load_state_dict(best_state); torch.save(model.state_dict(), output / f"head_{head}.pt")
        metrics.append({"head": head, "validation_binomial_nll": best, "epochs": epoch + 1})
    report = {"representation": args.representation,
              "train_suites": sorted(train_suites), "validation_suites": sorted(val_suites),
              "test_suites": sorted(test_suites), "heads": metrics,
              "input_dim": x_train.shape[1], "hidden_size": int(training["hidden_size"])}
    write_json(output / "training_report.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
