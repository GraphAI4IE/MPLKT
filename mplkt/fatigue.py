import math
from collections import defaultdict

import numpy as np


TIME_COLUMNS = ["elapsed_time", "timeTaken", "time_taken", "ms_first_response_time"]
ATTEMPT_COLUMNS = ["attempt_count", "attemptCount"]
QUESTION_COLUMNS = ["question_id", "problem_id", "assistment_id", "problemId", "assistmentId"]
ORDER_COLUMNS = ["timestamp", "startTime", "endTime", "order_id", "position", "action_num"]
REAL_TIME_COLUMNS = {"timestamp", "startTime", "endTime"}


class FatigueConfig(object):
    def __init__(self, args=None, **kwargs):
        source = vars(args) if args is not None else {}
        source.update(kwargs)
        self.k = int(source.get("fatigue_k", 5))
        self.time_window = float(source.get("fatigue_time_window", 1800.0))
        self.decay_lambda = float(source.get("fatigue_decay_lambda", 0.5))
        self.continuity_mode = source.get("fatigue_continuity_mode", "count")
        self.beta_d = float(source.get("fatigue_history_load_beta_d", source.get("fatigue_beta_d", 1.0)))
        self.beta_t = float(source.get("fatigue_history_load_beta_t", source.get("fatigue_beta_t", 1.0)))
        self.time_threshold_mode = source.get("fatigue_time_threshold_mode", "global")
        self.time_threshold_quantile = float(source.get("fatigue_time_threshold_quantile", 0.5))
        self.difficulty_alpha_smooth = float(source.get("difficulty_alpha_smooth", 1.0))
        self.difficulty_min_count_question = int(source.get("difficulty_min_count_question", 20))
        self.difficulty_min_count_skill = int(source.get("difficulty_min_count_skill", 20))
        self.time_min_count_question = int(source.get("fatigue_time_min_count_question", self.difficulty_min_count_question))
        self.time_min_count_skill = int(source.get("fatigue_time_min_count_skill", self.difficulty_min_count_skill))
        self.difficulty_stat_scope = source.get("difficulty_stat_scope", "train_only")
        self.time_load_clip = float(source.get("fatigue_time_load_clip", 5.0))
        self.missing_time_policy = source.get("fatigue_missing_time_policy", "zero")
        self.feature_mode = source.get("fatigue_feature_mode", "legacy")
        self.practice_load_time_source = source.get("practice_load_time_source", "current")
        if self.practice_load_time_source not in ("current", "previous"):
            raise ValueError("practice_load_time_source must be current or previous")
        windows = source.get("fatigue_history_windows", "300,1800,7200")
        if isinstance(windows, str):
            self.history_windows = [float(v.strip()) for v in windows.split(",") if v.strip()]
        else:
            self.history_windows = [float(v) for v in windows]
        if len(self.history_windows) != 3:
            raise ValueError("fatigue_history_windows must contain exactly three windows, e.g. 300,1800,7200")
        self.history_windows = sorted(max(v, 1.0) for v in self.history_windows)
        self.junyi_time_mode = source.get("junyi_fatigue_time_mode", "unit10")
        if self.junyi_time_mode not in ("seconds", "unit10"):
            raise ValueError("junyi_fatigue_time_mode must be seconds or unit10")
        self.time_unit_seconds = 1.0
        self.quantize_time_units = False

    def enable_quantized_time_units(self, unit_seconds):
        unit_seconds = float(unit_seconds)
        if unit_seconds <= 0.0:
            raise ValueError("unit_seconds must be positive")
        if self.quantize_time_units:
            if self.time_unit_seconds != unit_seconds:
                raise ValueError("fatigue time units were already configured with a different scale")
            return
        self.time_unit_seconds = unit_seconds
        self.quantize_time_units = True
        self.time_window = max(self.time_window / unit_seconds, 1.0)
        self.history_windows = [max(value / unit_seconds, 1.0) for value in self.history_windows]

    def to_time_units(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return np.nan
        if np.isnan(numeric) or not self.quantize_time_units:
            return numeric
        if numeric <= 0.0:
            return 0.0
        return float(max(1, int(math.floor(numeric / self.time_unit_seconds + 0.5))))


def first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def parse_numeric_series(series):
    return series.apply(lambda v: _parse_numeric(v))


def _parse_numeric(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            parsed = np.datetime64(value)
            if np.isnat(parsed):
                return np.nan
            return parsed.astype("datetime64[s]").astype(float)
        except (TypeError, ValueError):
            return np.nan


def build_fatigue_statistics(rows, config):
    stats = FatigueStatistics(config)
    stats.fit(rows)
    return stats


class FatigueStatistics(object):
    def __init__(self, config):
        self.config = config
        self.global_tau = 0.0
        self.global_weighted_log_time = 0.0
        self.skill_tau = {}
        self.question_tau = {}
        self.global_difficulty = 0.5
        self.skill_difficulty = {}
        self.question_difficulty = {}
        self.skill_counts = {}
        self.question_counts = {}

    def fit(self, rows):
        rows = [r for r in rows if r.get("valid", True)]
        self._fit_time_thresholds(rows)
        self._fit_difficulty(rows)
        self._fit_weighted_time(rows)

    def _fit_time_thresholds(self, rows):
        time_rows = [r for r in rows if not np.isnan(r["elapsed_time"])]
        log_times = np.array([math.log1p(max(0.0, r["elapsed_time"])) for r in time_rows], dtype=np.float32)
        if log_times.size:
            self.global_tau = float(np.quantile(log_times, self.config.time_threshold_quantile))
        else:
            self.global_tau = 0.0

        by_skill = defaultdict(list)
        by_question = defaultdict(list)
        for r in time_rows:
            log_time = math.log1p(max(0.0, r["elapsed_time"]))
            by_skill[r["skill"]].append(log_time)
            if r["question"] is not None:
                by_question[r["question"]].append(log_time)

        self.skill_tau = {
            k: float(np.quantile(v, self.config.time_threshold_quantile))
            for k, v in by_skill.items()
            if len(v) >= self.config.time_min_count_skill
        }
        self.question_tau = {
            k: float(np.quantile(v, self.config.time_threshold_quantile))
            for k, v in by_question.items()
            if len(v) >= self.config.time_min_count_question
        }

    def _fit_difficulty(self, rows):
        alpha = self.config.difficulty_alpha_smooth
        total = len(rows)
        correct = sum(float(r["correct"]) for r in rows)
        if total:
            self.global_difficulty = 1.0 - ((correct + alpha) / (total + 2.0 * alpha))
        else:
            self.global_difficulty = 0.5

        by_skill = defaultdict(lambda: [0.0, 0])
        by_question = defaultdict(lambda: [0.0, 0])
        for r in rows:
            by_skill[r["skill"]][0] += float(r["correct"])
            by_skill[r["skill"]][1] += 1
            if r["question"] is not None:
                by_question[r["question"]][0] += float(r["correct"])
                by_question[r["question"]][1] += 1

        self.skill_counts = {k: v[1] for k, v in by_skill.items()}
        self.question_counts = {k: v[1] for k, v in by_question.items()}
        self.skill_difficulty = {
            k: 1.0 - ((v[0] + alpha) / (v[1] + 2.0 * alpha))
            for k, v in by_skill.items()
            if v[1] >= self.config.difficulty_min_count_skill
        }
        self.question_difficulty = {
            k: 1.0 - ((v[0] + alpha) / (v[1] + 2.0 * alpha))
            for k, v in by_question.items()
            if v[1] >= self.config.difficulty_min_count_question
        }

    def _fit_weighted_time(self, rows):
        numerator = 0.0
        denominator = 0.0
        for r in rows:
            if np.isnan(r["elapsed_time"]):
                continue
            difficulty = min(max(self.difficulty(r["question"], r["skill"]), 0.0), 1.0)
            weight = _difficulty_weight(difficulty)
            numerator += weight * math.log1p(max(0.0, r["elapsed_time"]))
            denominator += weight
        self.global_weighted_log_time = numerator / denominator if denominator > 0.0 else self.global_tau

    def time_threshold(self, question, skill):
        mode = self.config.time_threshold_mode
        if mode == "question" and question in self.question_tau:
            return self.question_tau[question]
        if mode in ("question", "skill") and skill in self.skill_tau:
            return self.skill_tau[skill]
        return self.global_tau

    def difficulty(self, question, skill):
        if question in self.question_difficulty:
            return self.question_difficulty[question]
        if skill in self.skill_difficulty:
            return self.skill_difficulty[skill]
        return self.global_difficulty


def build_sequence_fatigue(sequence_rows, stats, config, has_real_time):
    if not sequence_rows:
        return [], []
    if config.feature_mode == "personalized":
        return _build_sequence_fatigue_personalized(sequence_rows, stats, config, has_real_time)
    weights = [math.exp(-config.decay_lambda * j) for j in range(1, config.k + 1)]
    weight_sum = max(sum(weights), 1e-8)
    load_norm = max(config.beta_d + config.beta_t, 1e-8)

    c_values = []
    t_values = []
    d_values = []
    contexts = []

    for i, row in enumerate(sequence_rows):
        if np.isnan(row["elapsed_time"]):
            if config.missing_time_policy == "error":
                raise ValueError(
                    "Fatigue features are enabled but no elapsed-time column is available. "
                    "Provide elapsed_time/timeTaken/ms_first_response_time or use --fatigue-missing-time-policy zero."
                )
            elapsed = 0.0
        else:
            elapsed = max(0.0, float(row["elapsed_time"]))

        tau = stats.time_threshold(row["question"], row["skill"])
        # The current response duration is observed after this interaction.
        raw_t_load = max(0.0, math.log1p(elapsed) - tau)
        t_load = min(raw_t_load, config.time_load_clip) / max(config.time_load_clip, 1e-8)
        difficulty = min(max(stats.difficulty(row["question"], row["skill"]), 0.0), 1.0)

        t_values.append(t_load)
        d_values.append(difficulty)
        sequence_time = 0.0 if np.isnan(row["sequence_time"]) else float(row["sequence_time"])
        time_valid = 1.0 if has_real_time and not np.isnan(row["sequence_time"]) else 0.0
        contexts.append([
            elapsed,
            0.0 if np.isnan(row["attempt_count"]) else float(row["attempt_count"]),
            difficulty,
            sequence_time,
            time_valid,
            tau,
        ])

        c_raw = 0.0
        for j, weight in enumerate(weights, start=1):
            hist_idx = i - j
            if hist_idx < 0:
                break
            if has_real_time:
                cur_time = row["sequence_time"]
                hist_time = sequence_rows[hist_idx]["sequence_time"]
                gap = config.to_time_units(abs(cur_time - hist_time))
                if np.isnan(cur_time) or np.isnan(hist_time) or gap >= config.time_window:
                    continue
            hist_load = 1.0
            if config.continuity_mode == "load":
                hist_load = config.beta_d * d_values[hist_idx] + config.beta_t * t_values[hist_idx]
            c_raw += weight * hist_load
        if config.continuity_mode == "load":
            c_value = min(c_raw / (weight_sum * load_norm), 1.0)
        else:
            c_value = min(c_raw / weight_sum, 1.0)
        c_values.append(c_value)

    features = []
    for c_value, t_load, difficulty in zip(c_values, t_values, d_values):
        features.append([
            c_value,
            t_load,
            difficulty,
            c_value * t_load,
            c_value * difficulty,
            t_load * difficulty,
            c_value * t_load * difficulty,
        ])
    return features, contexts


def _difficulty_weight(difficulty):
    return 0.1 + min(max(float(difficulty), 0.0), 1.0)


def _weighted_mean(values, weights, default=0.0):
    denominator = sum(weights)
    if denominator <= 1e-8:
        return default
    return sum(v * w for v, w in zip(values, weights)) / denominator


def _softplus(value):
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _normalized_load(value, clip):
    return min(max(value, 0.0), clip) / max(clip, 1e-8)


def _build_sequence_fatigue_personalized(sequence_rows, stats, config, has_real_time):
    log_times = []
    current_loads = []
    d_values = []
    times = []
    contexts = []

    for row in sequence_rows:
        if np.isnan(row["elapsed_time"]):
            if config.missing_time_policy == "error":
                raise ValueError(
                    "Fatigue features are enabled but no elapsed-time column is available. "
                    "Provide elapsed_time/timeTaken/ms_first_response_time or use --fatigue-missing-time-policy zero."
                )
            elapsed = 0.0
        else:
            elapsed = max(0.0, float(row["elapsed_time"]))

        log_time = math.log1p(elapsed)
        difficulty = min(max(stats.difficulty(row["question"], row["skill"]), 0.0), 1.0)
        tau = stats.time_threshold(row["question"], row["skill"])
        sequence_time = 0.0 if np.isnan(row["sequence_time"]) else float(row["sequence_time"])
        time_valid = 1.0 if has_real_time and not np.isnan(row["sequence_time"]) else 0.0

        log_times.append(log_time)
        d_values.append(difficulty)
        times.append(sequence_time)
        contexts.append([
            elapsed,
            0.0 if np.isnan(row["attempt_count"]) else float(row["attempt_count"]),
            difficulty,
            sequence_time,
            time_valid,
            tau,
        ])

    features = []
    global_mean = float(stats.global_weighted_log_time)
    long_window = max(config.history_windows[-1], config.time_window, 1.0)
    for i, row in enumerate(sequence_rows):
        history = list(range(i))
        hist_weights = [_difficulty_weight(d_values[j]) for j in history]
        user_mean = _weighted_mean(
            [log_times[j] for j in history],
            hist_weights,
            default=global_mean,
        )

        recent_start = max(0, i - config.k)
        recent = list(range(recent_start, i))
        recent_weights = []
        for j in recent:
            distance = i - j
            recent_weights.append(_difficulty_weight(d_values[j]) * math.exp(-config.decay_lambda * distance))
        recent_mean = _weighted_mean(
            [log_times[j] for j in recent],
            recent_weights,
            default=user_mean,
        )

        tau = stats.time_threshold(row["question"], row["skill"])
        tau_user = tau + (user_mean - global_mean)
        current_over_user = log_times[i] - user_mean
        current_over_tau = log_times[i] - tau_user
        current_load = _normalized_load(_softplus(current_over_tau), config.time_load_clip)
        current_loads.append(current_load)

        recent_shift = max(0.0, recent_mean - user_mean)
        recent_shift_load = _normalized_load(recent_shift, config.time_load_clip)
        user_global_gap = _normalized_load(abs(user_mean - global_mean), config.time_load_clip)
        current_user_gap = _normalized_load(_softplus(current_over_user), config.time_load_clip)

        window_work_values = []
        window_weights = []
        cur_time = times[i]
        if has_real_time and not np.isnan(row["sequence_time"]):
            for j in history:
                hist_time = times[j]
                if np.isnan(hist_time):
                    continue
                gap = config.to_time_units(cur_time - hist_time)
                if 0.0 < gap <= config.time_window:
                    time_weight = math.exp(-config.decay_lambda * gap / max(config.time_window, 1.0))
                    window_weights.append(_difficulty_weight(d_values[j]) * time_weight)
                    window_work_values.append(1.0 + current_loads[j])
            if i == 0:
                gap_value = 1.0
            else:
                prev_time = times[i - 1]
                if np.isnan(prev_time):
                    gap_value = 1.0
                else:
                    previous_gap = config.to_time_units(max(0.0, cur_time - prev_time))
                    gap_value = min(math.log1p(previous_gap) / math.log1p(long_window), 1.0)
        else:
            gap_value = 0.0
        window_workload = _normalized_load(
            sum(v * w for v, w in zip(window_work_values, window_weights)),
            max(config.k, 1) * (1.0 + config.time_load_clip),
        )

        features.append([
            current_user_gap,
            current_load,
            d_values[i],
            user_global_gap,
            recent_shift_load,
            window_workload,
            gap_value,
        ])
    if config.practice_load_time_source == "previous":
        instantaneous = [[0.0, 0.0]] + [feature[:2] for feature in features[:-1]]
        features = [instantaneous[i] + feature[2:] for i, feature in enumerate(features)]
    return features, contexts
