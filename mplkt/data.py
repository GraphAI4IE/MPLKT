import csv
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .fatigue import FatigueConfig, build_fatigue_statistics, build_sequence_fatigue


PAD_LONG = -1


@dataclass
class DatasetBundle:
    concept_num: int
    problem_num: int
    combo_num: int
    qs_adj: torch.Tensor
    problem_difficulty: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    problem_to_idx: Dict[str, int]
    skill_to_idx: Dict[str, int]
    combo_to_idx: Dict[Tuple[int, ...], int]
    practice_load_metadata: Dict[str, object]


class MPLKTDataset(Dataset):
    def __init__(self, chunks: Sequence[dict]):
        self.chunks = list(chunks)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


def collate_fn(batch):
    keys_1d_long = ["problem_ids", "combo_ids", "correct"]
    out = {}
    for key in keys_1d_long:
        vals = [torch.as_tensor(item[key], dtype=torch.long) for item in batch]
        out[key] = pad_sequence(vals, batch_first=True, padding_value=PAD_LONG)

    vals = [torch.as_tensor(item["mask"], dtype=torch.bool) for item in batch]
    out["mask"] = pad_sequence(vals, batch_first=True, padding_value=False)

    for key in ["fatigue", "difficulty"]:
        vals = [torch.as_tensor(item[key], dtype=torch.float32) for item in batch]
        out[key] = pad_sequence(vals, batch_first=True, padding_value=0.0)

    vals = [torch.as_tensor(item["multi_skill_mask"], dtype=torch.float32) for item in batch]
    out["multi_skill_mask"] = pad_sequence(vals, batch_first=True, padding_value=0.0)

    hist_idx = [torch.as_tensor(item["hist_idx"], dtype=torch.long) for item in batch]
    hist_len = [torch.as_tensor(item["hist_len"], dtype=torch.long) for item in batch]
    max_len = max(x.size(0) for x in hist_idx)
    hist_k = hist_idx[0].size(1)
    padded_idx = []
    padded_len = []
    for idx, length in zip(hist_idx, hist_len):
        if idx.size(0) < max_len:
            pad_rows = torch.full((max_len - idx.size(0), hist_k), PAD_LONG, dtype=torch.long)
            idx = torch.cat([idx, pad_rows], dim=0)
            length = torch.cat([length, torch.zeros(max_len - length.size(0), dtype=torch.long)], dim=0)
        padded_idx.append(idx)
        padded_len.append(length)
    out["hist_idx"] = torch.stack(padded_idx, dim=0)
    out["hist_len"] = torch.stack(padded_len, dim=0)
    return out


def _split_skills(value) -> List[str]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).replace("|", ";").split(";")]
    return [p for p in parts if p]


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        required = {"user_id", "skill_id", "correct"}
        missing = required - columns
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        for line_no, row in enumerate(reader):
            user_id = row.get("user_id")
            skill_id = row.get("skill_id")
            if user_id is None or skill_id is None:
                continue
            y = _to_float(row.get("correct"), default=float("nan"))
            if y not in (0.0, 1.0):
                continue
            skills = _split_skills(skill_id)
            if not skills:
                continue
            order = _to_float(row.get("order_id", row.get("timestamp", line_no)), default=float(line_no))
            has_real_time = "timestamp" in columns and row.get("timestamp") not in (None, "")
            timestamp = _to_float(row.get("timestamp"), default=float("nan"))
            elapsed = _to_float(row.get("elapsed_time"), default=float("nan"))
            attempt_count = _to_float(row.get("attempt_count"), default=float("nan"))
            rows.append(
                {
                    "user_id": str(user_id),
                    "problem_id": str(row.get("problem_id") or skill_id),
                    "correct": int(y),
                    "skill_list": skills,
                    "order_id": order,
                    "sequence_time": timestamp,
                    "has_real_time": has_real_time,
                    "elapsed_time": elapsed if np.isnan(elapsed) else max(elapsed, 0.0),
                    "attempt_count": attempt_count,
                    "source_csv_line": int(line_no + 2),
                }
            )
    rows.sort(key=lambda r: (r["user_id"], r["order_id"]))
    user_position = defaultdict(int)
    for row in rows:
        row["user_position"] = int(user_position[row["user_id"]])
        user_position[row["user_id"]] += 1
    return rows


def _build_history_indices(skill_lists: List[List[int]], valid_len: int, history_k: int):
    hist_idx = np.full((valid_len, history_k), PAD_LONG, dtype=np.int64)
    hist_len = np.zeros(valid_len, dtype=np.int64)
    recent_by_skill = defaultdict(lambda: deque(maxlen=history_k))
    duplicate_removed = 0
    for t in range(valid_len):
        candidates = []
        for s in skill_lists[t]:
            candidates.extend(recent_by_skill[s])
        before = len(candidates)
        candidates = sorted(set(candidates))
        duplicate_removed += before - len(candidates)
        candidates = candidates[-history_k:]
        hist_len[t] = len(candidates)
        if candidates:
            hist_idx[t, : len(candidates)] = candidates
        for s in skill_lists[t]:
            recent_by_skill[s].append(t)
    return hist_idx, hist_len, duplicate_removed


def _fatigue_rows(records):
    rows = []
    for r in records:
        rows.append(
            {
                "skill": int(r["skill_indices"][0]),
                "question": int(r["problem_idx"]),
                "correct": int(r["correct"]),
                "elapsed_time": float(r.get("elapsed_time", np.nan)),
                "attempt_count": float(r.get("attempt_count", np.nan)),
                "sequence_time": float(r.get("sequence_time", np.nan)),
                "valid": True,
            }
        )
    return rows


def _make_chunks(rows, max_seq_len, min_seq_len):
    chunks = []
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)
    for user_id in sorted(by_user):
        records = by_user[user_id]
        for start in range(0, len(records), max_seq_len):
            part = records[start : start + max_seq_len]
            if len(part) < min_seq_len:
                continue
            chunks.append(part)
    return chunks


def _materialize_chunks(raw_chunks, concept_num, fatigue_stats, fatigue_config, history_k, has_real_time):
    chunks = []
    dup_removed = 0
    for records in raw_chunks:
        length = len(records)
        problems = [int(r["problem_idx"]) for r in records]
        combos = [int(r["combo_id"]) for r in records]
        correct = [int(r["correct"]) for r in records]
        skill_lists = [list(r["skill_indices"]) for r in records]
        fatigue_rows = _fatigue_rows(records)
        fatigue_features, _ = build_sequence_fatigue(
            fatigue_rows, fatigue_stats, fatigue_config, has_real_time
        )
        fatigue_features = np.asarray(fatigue_features, dtype=np.float32)
        difficulties = fatigue_features[:, 2].astype(np.float32) if fatigue_features.size else np.zeros(length, dtype=np.float32)
        mask = np.ones(length, dtype=np.bool_)
        multi_skill = np.zeros((length, concept_num), dtype=np.float32)
        for i, skills in enumerate(skill_lists):
            multi_skill[i, skills] = 1.0
        hist_idx, hist_len, dup = _build_history_indices(skill_lists, length, history_k)
        dup_removed += dup
        chunks.append(
            {
                "problem_ids": np.asarray(problems, dtype=np.int64),
                "combo_ids": np.asarray(combos, dtype=np.int64),
                "correct": np.asarray(correct, dtype=np.int64),
                "mask": mask,
                "multi_skill_mask": multi_skill,
                "difficulty": np.asarray(difficulties, dtype=np.float32).reshape(length, 1),
                "fatigue": fatigue_features,
                "hist_idx": hist_idx,
                "hist_len": hist_len,
            }
        )
    return chunks, dup_removed


def load_mplkt_dataset(
    file_path: str,
    batch_size: int,
    max_seq_len: int = 200,
    min_seq_len: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    history_k: int = 20,
    seed: int = 42,
    shuffle_train: bool = True,
    skill_update_mode: str = "combo",
    fatigue_time_threshold_quantile: float = 0.5,
    fatigue_decay_lambda: float = 0.5,
    practice_load_time_source: str = "current",
) -> DatasetBundle:
    if skill_update_mode not in {"combo", "first_skill"}:
        raise ValueError("skill_update_mode must be combo or first_skill")
    rows = _normalize_rows(file_path)
    if skill_update_mode == "first_skill":
        for row in rows:
            row["skill_list"] = row["skill_list"][:1]
    fatigue_config = FatigueConfig(
        fatigue_feature_mode="personalized",
        fatigue_time_threshold_quantile=fatigue_time_threshold_quantile,
        fatigue_decay_lambda=fatigue_decay_lambda,
        practice_load_time_source=practice_load_time_source,
    )
    is_junyi = os.path.basename(file_path).lower().startswith("junyi")
    if is_junyi and fatigue_config.junyi_time_mode == "unit10":
        fatigue_config.enable_quantized_time_units(10.0)
        for row in rows:
            row["elapsed_time"] = fatigue_config.to_time_units(row.get("elapsed_time", np.nan))

    skill_values = sorted({s for row in rows for s in row["skill_list"]})
    problem_values = sorted({row["problem_id"] for row in rows})
    skill_to_idx = {v: i for i, v in enumerate(skill_values)}
    problem_to_idx = {v: i for i, v in enumerate(problem_values)}

    for row in rows:
        row["skill_indices"] = sorted({skill_to_idx[v] for v in row["skill_list"]})
        row["problem_idx"] = problem_to_idx[row["problem_id"]]

    combo_keys = sorted({tuple(row["skill_indices"]) for row in rows})
    combo_to_idx = {v: i for i, v in enumerate(combo_keys)}
    for row in rows:
        row["combo_id"] = combo_to_idx[tuple(row["skill_indices"])]

    concept_num = len(skill_to_idx)
    problem_num = len(problem_to_idx)
    combo_num = len(combo_to_idx)

    raw_chunks = _make_chunks(rows, max_seq_len, min_seq_len)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(raw_chunks), generator=generator).numpy()
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    train_idx = set(indices[:train_end])
    val_idx = set(indices[train_end:val_end])

    train_rows = []
    for idx, records in enumerate(raw_chunks):
        if idx in train_idx:
            train_rows.extend(records)
    fatigue_stats = build_fatigue_statistics(_fatigue_rows(train_rows), fatigue_config)
    has_real_time = any(bool(row.get("has_real_time")) and not np.isnan(float(row.get("sequence_time", np.nan))) for row in rows)

    chunks, _ = _materialize_chunks(raw_chunks, concept_num, fatigue_stats, fatigue_config, history_k, has_real_time)
    train_chunks = [chunks[i] for i in range(len(chunks)) if i in train_idx]
    val_chunks = [chunks[i] for i in range(len(chunks)) if i in val_idx]
    test_chunks = [chunks[i] for i in range(len(chunks)) if i not in train_idx and i not in val_idx]

    qs_adj = torch.zeros(problem_num, concept_num, dtype=torch.float32)
    for row in rows:
        qs_adj[int(row["problem_idx"]), list(row["skill_indices"])] = 1.0
    problem_difficulty = torch.zeros(problem_num, dtype=torch.float32)
    first_skill_by_problem = {}
    for row in rows:
        first_skill_by_problem.setdefault(int(row["problem_idx"]), int(row["skill_indices"][0]))
    for problem_idx in range(problem_num):
        problem_difficulty[problem_idx] = float(
            fatigue_stats.difficulty(problem_idx, first_skill_by_problem[problem_idx])
        )

    train_loader = DataLoader(MPLKTDataset(train_chunks), batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
    val_loader = DataLoader(MPLKTDataset(val_chunks), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(MPLKTDataset(test_chunks), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return DatasetBundle(
        concept_num=concept_num,
        problem_num=problem_num,
        combo_num=combo_num,
        qs_adj=qs_adj,
        problem_difficulty=problem_difficulty,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        problem_to_idx=problem_to_idx,
        skill_to_idx=skill_to_idx,
        combo_to_idx=combo_to_idx,
        practice_load_metadata={
            "feature_mode": fatigue_config.feature_mode,
            "time_threshold_mode": fatigue_config.time_threshold_mode,
            "time_threshold_quantile": float(fatigue_config.time_threshold_quantile),
            "time_source": fatigue_config.practice_load_time_source,
            "global_time_threshold": float(fatigue_stats.global_tau),
            "time_window": float(fatigue_config.time_window),
            "decay_lambda": float(fatigue_config.decay_lambda),
            "time_load_clip": float(fatigue_config.time_load_clip),
            "time_unit_seconds": float(fatigue_config.time_unit_seconds),
            "quantize_time_units": bool(fatigue_config.quantize_time_units),
            "has_real_time": bool(has_real_time),
        },
    )
