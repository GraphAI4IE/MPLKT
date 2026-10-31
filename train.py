import argparse
from collections import defaultdict
import hashlib
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from mplkt.data import load_mplkt_dataset
from mplkt.metrics import auc_acc, collect_masked, masked_bce
from mplkt.model import MPLKTModel


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser(description="Train clean MPLKT model.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--save-dir", default="logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--min-seq-len", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr-decay", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")

    parser.add_argument("--hid-dim", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--q-graph-dim", type=int, default=64)
    parser.add_argument("--qs-gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fatigue-dim", type=int, default=16)
    parser.add_argument("--fatigue-hidden-dim", type=int, default=64)
    parser.add_argument("--fatigue-time-threshold-quantile", type=float, default=0.5)
    parser.add_argument("--fatigue-decay-lambda", type=float, default=0.5)
    parser.add_argument("--practice-load-time-source", choices=["current", "previous"], default="current")
    parser.add_argument("--history-k", type=int, default=20)
    parser.add_argument("--history-gru-hidden-dim", type=int, default=32)
    parser.add_argument("--lambda-cl", type=float, default=0.01)
    parser.add_argument("--contrastive-neg-k", type=int, default=10)
    parser.add_argument("--contrastive-tau", type=float, default=0.2)
    parser.add_argument("--use-fatigue", type=str2bool, default=True)
    parser.add_argument("--use-qs-structural-learning", type=str2bool, default=True)
    parser.add_argument("--skill-update-mode", choices=["combo", "first_skill"], default="combo")
    return parser


PRACTICE_LOAD_FEATURE_NAMES = [
    "current_user_gap",
    "instantaneous_load_input",
    "difficulty",
    "user_global_gap",
    "recent_shift_load",
    "window_workload",
    "rest_gap",
]


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "near_zero_ratio": float(np.mean(values <= 0.05)),
        "near_one_ratio": float(np.mean(values >= 0.95)),
    }


def json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_loss(model, batch, lambda_cl, collect_load_stats=False):
    model_output = model(
        batch,
        return_cl=lambda_cl > 0,
        return_load_diagnostics=collect_load_stats,
    )
    if collect_load_stats:
        pred, cl_loss, load_diagnostics = model_output
    else:
        pred, cl_loss = model_output
        load_diagnostics = None
    correct = batch["correct"].to(pred.device)[:, 1:]
    mask = batch["mask"].to(pred.device)[:, 1:]
    kt_loss = masked_bce(pred, correct, mask)
    total = kt_loss + lambda_cl * cl_loss
    return total, kt_loss.detach(), cl_loss.detach(), pred.detach(), correct.detach(), mask.detach(), load_diagnostics


def run_epoch(model, loader, optimizer, lambda_cl, train, collect_load_stats=False):
    model.train(train)
    losses = []
    kt_losses = []
    cl_losses = []
    pred_parts = []
    target_parts = []
    load_parts = defaultdict(list) if collect_load_stats else None
    feature_parts = defaultdict(list) if collect_load_stats else None
    with torch.set_grad_enabled(train):
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            total, kt_loss, cl_loss, pred, correct, mask, load_diagnostics = batch_loss(
                model, batch, lambda_cl, collect_load_stats=collect_load_stats
            )
            if train:
                total.backward()
                optimizer.step()
            losses.append(float(total.detach().cpu()))
            kt_losses.append(float(kt_loss.cpu()))
            cl_losses.append(float(cl_loss.cpu()))
            p, y = collect_masked(pred, correct, mask)
            if len(y):
                pred_parts.append(p)
                target_parts.append(y)
            if collect_load_stats:
                valid_mask = mask.bool()
                for name, tensor in load_diagnostics.items():
                    values = tensor[valid_mask].detach().cpu().numpy()
                    if values.size:
                        load_parts[name].append(values)
                feature_values = batch["fatigue"][:, 1:, :]
                feature_mask = batch["mask"][:, 1:].bool()
                for feature_index, feature_name in enumerate(PRACTICE_LOAD_FEATURE_NAMES):
                    values = feature_values[:, :, feature_index][feature_mask].numpy()
                    if values.size:
                        feature_parts[feature_name].append(values)
    if pred_parts:
        pred_values = np.concatenate(pred_parts)
        target_values = np.concatenate(target_parts)
    else:
        pred_values = np.asarray([])
        target_values = np.asarray([])
    auc, acc = auc_acc(pred_values, target_values)
    stats = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "kt_loss": float(np.mean(kt_losses)) if kt_losses else float("nan"),
        "cl_loss": float(np.mean(cl_losses)) if cl_losses else float("nan"),
        "auc": auc,
        "acc": acc,
    }
    if collect_load_stats:
        stats["practice_load"] = {
            "valid_interactions": int(target_values.size),
            "model_outputs": {
                name: summarize_values(np.concatenate(parts) if parts else np.asarray([]))
                for name, parts in load_parts.items()
            },
            "input_features": {
                name: summarize_values(np.concatenate(parts) if parts else np.asarray([]))
                for name, parts in feature_parts.items()
            },
        }
    return stats


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    effective_lambda_cl = args.lambda_cl if args.use_qs_structural_learning else 0.0
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dataset_path = os.path.join(args.data_dir, args.data_file)
    run_name = "MPLKT-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")
    ckpt_path = os.path.join(save_dir, "MPLKT.pt")

    bundle = load_mplkt_dataset(
        dataset_path,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        history_k=args.history_k,
        seed=args.seed,
        shuffle_train=args.shuffle,
        skill_update_mode=args.skill_update_mode,
        fatigue_time_threshold_quantile=args.fatigue_time_threshold_quantile,
        fatigue_decay_lambda=args.fatigue_decay_lambda,
        practice_load_time_source=args.practice_load_time_source,
    )

    model = MPLKTModel(
        problem_num=bundle.problem_num,
        concept_num=bundle.concept_num,
        combo_num=bundle.combo_num,
        qs_adj=bundle.qs_adj,
        problem_difficulty=bundle.problem_difficulty,
        hid_dim=args.hid_dim,
        emb_dim=args.emb_dim,
        q_dim=args.q_graph_dim,
        fatigue_dim=args.fatigue_dim,
        fatigue_hidden_dim=args.fatigue_hidden_dim,
        history_hidden_dim=args.history_gru_hidden_dim,
        dropout=args.dropout,
        qs_gnn_layers=args.qs_gnn_layers,
        use_fatigue=args.use_fatigue,
        skill_update_mode=args.skill_update_mode,
        use_qs_structural_learning=args.use_qs_structural_learning,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"command: {' '.join(sys.argv)}\n")
        log.write(f"args: {json.dumps(vars(args), ensure_ascii=False, sort_keys=True)}\n")
        log.write(
            "qs_structure: "
            f"enabled={args.use_qs_structural_learning} "
            f"effective_layers={args.qs_gnn_layers if args.use_qs_structural_learning else 0} "
            f"beta_mode={'static_attention' if args.use_qs_structural_learning else 'mean'} "
            f"effective_lambda_cl={effective_lambda_cl}\n"
        )
        log.write(
            "data: "
            f"problems={bundle.problem_num} skills={bundle.concept_num} combos={bundle.combo_num} "
            f"train_batches={len(bundle.train_loader)} val_batches={len(bundle.val_loader)} "
            f"test_batches={len(bundle.test_loader)}\n"
        )

    best_auc = -1.0
    best_epoch = -1
    best_val_stats = None
    bad_epochs = 0
    for epoch in range(args.epochs):
        model.train(True)
        model.refresh_static_cache(track_static_grad=True)
        train_stats = run_epoch(model, bundle.train_loader, optimizer, effective_lambda_cl, True)
        # Transfer accumulated static-encoder gradients once per epoch.
        optimizer.zero_grad(set_to_none=True)
        if model.backward_static_encoder():
            optimizer.step()
        model.eval()
        model.refresh_static_cache(track_static_grad=False)
        val_stats = run_epoch(model, bundle.val_loader, optimizer, effective_lambda_cl, False)
        line = (
            f"Epoch {epoch:04d} | "
            f"loss_train {train_stats['loss']:.6f} auc_train {train_stats['auc']:.6f} acc_train {train_stats['acc']:.6f} | "
            f"loss_val {val_stats['loss']:.6f} auc_val {val_stats['auc']:.6f} acc_val {val_stats['acc']:.6f}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(line + "\n")
            log.write(f"  train_detail: {train_stats}\n")
            log.write(f"  val_detail: {val_stats}\n")
        is_better = best_epoch < 0 or (not np.isnan(val_stats["auc"]) and val_stats["auc"] > best_auc)
        if is_better:
            best_auc = val_stats["auc"]
            best_epoch = epoch
            best_val_stats = dict(val_stats)
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "problem_to_idx": bundle.problem_to_idx,
                    "skill_to_idx": bundle.skill_to_idx,
                    "combo_to_idx": {str(k): v for k, v in bundle.combo_to_idx.items()},
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                    "best_val_stats": best_val_stats,
                    "effective_config": {
                        "qs_structural_learning": bool(args.use_qs_structural_learning),
                        "qs_gnn_layers": int(args.qs_gnn_layers if args.use_qs_structural_learning else 0),
                        "qs_weight_mode": "static_attention" if args.use_qs_structural_learning else "mean",
                        "lambda_cl": float(effective_lambda_cl),
                    },
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
        scheduler.step()
        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            break

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.refresh_static_cache(track_static_grad=False)
    test_stats = run_epoch(
        model,
        bundle.test_loader,
        optimizer,
        effective_lambda_cl,
        False,
        collect_load_stats=True,
    )
    practice_load_stats = test_stats.pop("practice_load")
    test_line = (
        f"test | loss {test_stats['loss']:.6f} auc {test_stats['auc']:.6f} "
        f"acc {test_stats['acc']:.6f} | best_epoch {best_epoch} best_val_auc {best_auc:.6f}"
    )
    print(test_line)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(test_line + "\n")

    sensitivity_record = {
        "schema_version": 1,
        "dataset": args.data_file,
        "dataset_path": os.path.normpath(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "run_directory": os.path.normpath(save_dir),
        "command": " ".join(sys.argv),
        "arguments": vars(args),
        "seed": int(args.seed),
        "time_threshold_quantile": float(args.fatigue_time_threshold_quantile),
        "data": {
            "problems": int(bundle.problem_num),
            "skills": int(bundle.concept_num),
            "skill_combinations": int(bundle.combo_num),
            "train_chunks": int(len(bundle.train_loader.dataset)),
            "validation_chunks": int(len(bundle.val_loader.dataset)),
            "test_chunks": int(len(bundle.test_loader.dataset)),
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "learning_rate": float(args.lr),
            "best_epoch": int(best_epoch),
            "best_val_auc": json_number(best_auc),
            "effective_qs_structural_learning": bool(args.use_qs_structural_learning),
            "effective_qs_gnn_layers": int(args.qs_gnn_layers if args.use_qs_structural_learning else 0),
            "effective_qs_weight_mode": "static_attention" if args.use_qs_structural_learning else "mean",
            "effective_lambda_cl": float(effective_lambda_cl),
        },
        "validation": {
            name: json_number(value)
            for name, value in (checkpoint.get("best_val_stats") or {}).items()
        },
        "test": {
            "loss": json_number(test_stats["loss"]),
            "kt_loss": json_number(test_stats["kt_loss"]),
            "cl_loss": json_number(test_stats["cl_loss"]),
            "auc": json_number(test_stats["auc"]),
            "acc": json_number(test_stats["acc"]),
        },
        "practice_load_config": bundle.practice_load_metadata,
        "practice_load_statistics": practice_load_stats,
    }
    sensitivity_path = os.path.join(save_dir, "practice_load_sensitivity.json")
    with open(sensitivity_path, "w", encoding="utf-8") as record_file:
        json.dump(sensitivity_record, record_file, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
