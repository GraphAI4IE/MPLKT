import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def masked_softmax(logits, mask, dim=-1):
    logits = logits.masked_fill(~mask.bool(), -1e9)
    return torch.softmax(logits, dim=dim)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateUpdateMLP(nn.Module):
    """Two-layer MLP with batch normalization for state-update messages."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.1)
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            return inputs
        if inputs.dim() == 3:
            flat = inputs.reshape(inputs.size(0) * inputs.size(1), -1)
            flat = self.norm(flat)
            return flat.view(inputs.size(0), inputs.size(1), -1)
        return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """Apply skill-specific erase and add operations before the GRU update."""

    def __init__(self, feature_dim, concept_num, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        erase_gate = torch.sigmoid(self.erase(x))
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))
        return tmp_x + self.weight.unsqueeze(dim=1) * add_feat


class VectorFatigueModule(nn.Module):
    """Encode instantaneous and continuous practice load."""

    def __init__(self, input_dim=7, hidden_dim=64, fatigue_dim=16, dropout=0.1):
        super().__init__()
        if fatigue_dim < 2:
            raise ValueError("fatigue_dim must be at least 2")
        self.current_dim = max(1, fatigue_dim // 2)
        self.continuous_dim = fatigue_dim - self.current_dim
        self.current_input_dim = min(3, input_dim)
        self.continuous_input_dim = max(input_dim - self.current_input_dim, 1)
        self.current_net = nn.Sequential(
            nn.Linear(self.current_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.current_dim),
        )
        self.continuous_net = nn.Sequential(
            nn.Linear(self.continuous_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.continuous_dim),
        )
        self.fatigue_norm = nn.LayerNorm(fatigue_dim)
        self.score = nn.Linear(fatigue_dim, 1)
        self.current_score = nn.Linear(self.current_dim, 1)
        self.continuous_score = nn.Linear(self.continuous_dim, 1)

    def forward(self, fatigue_features, mask=None):
        current_input = fatigue_features[:, :self.current_input_dim]
        continuous_input = fatigue_features[:, self.current_input_dim:]
        if continuous_input.shape[-1] == 0:
            continuous_input = torch.zeros(
                (fatigue_features.shape[0], self.continuous_input_dim),
                device=fatigue_features.device,
                dtype=fatigue_features.dtype,
            )
        fatigue_current = self.current_net(current_input)
        fatigue_continuous = self.continuous_net(continuous_input)
        fatigue_vec = self.fatigue_norm(torch.cat((fatigue_current, fatigue_continuous), dim=-1))
        f_score = torch.sigmoid(self.score(fatigue_vec))
        current_score = torch.sigmoid(self.current_score(fatigue_current))
        continuous_score = torch.sigmoid(self.continuous_score(fatigue_continuous))
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            fatigue_vec = fatigue_vec * mask
            f_score = f_score * mask
            current_score = current_score * mask
            continuous_score = continuous_score * mask
        return fatigue_vec, f_score, current_score, continuous_score


class QSGraphEncoder(nn.Module):
    def __init__(self, graph_dim, hidden_dim=64, layers=2, dropout=0.1,
                 use_layernorm=True, use_residual=True):
        super().__init__()
        self.layers = int(layers)
        self.use_residual = use_residual
        self.s2q = nn.ModuleList([nn.Linear(graph_dim, graph_dim) for _ in range(self.layers)])
        self.q2s = nn.ModuleList([nn.Linear(graph_dim, graph_dim) for _ in range(self.layers)])
        self.q_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(graph_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, graph_dim),
            )
            for _ in range(self.layers)
        ])
        self.s_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(graph_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, graph_dim),
            )
            for _ in range(self.layers)
        ])
        self.q_norm = nn.ModuleList([
            nn.LayerNorm(graph_dim) if use_layernorm else nn.Identity()
            for _ in range(self.layers)
        ])
        self.s_norm = nn.ModuleList([
            nn.LayerNorm(graph_dim) if use_layernorm else nn.Identity()
            for _ in range(self.layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_nodes, s_nodes, qs_adj):
        if self.layers <= 0:
            return q_nodes, s_nodes
        adj = qs_adj.float()
        q_deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        s_deg = adj.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
        z_q, z_s = q_nodes, s_nodes
        for layer in range(self.layers):
            q_prev, s_prev = z_q, z_s
            m_q = torch.matmul(adj, self.s2q[layer](s_prev)) / q_deg
            m_s = torch.matmul(adj.transpose(0, 1), self.q2s[layer](q_prev)) / s_deg
            q_delta = self.dropout(self.q_update[layer](torch.cat((q_prev, m_q), dim=-1)))
            s_delta = self.dropout(self.s_update[layer](torch.cat((s_prev, m_s), dim=-1)))
            z_q = q_prev + q_delta if self.use_residual else q_delta
            z_s = s_prev + s_delta if self.use_residual else s_delta
            z_q = self.q_norm[layer](z_q)
            z_s = self.s_norm[layer](z_s)
        return z_q, z_s


class StaticQSEdgeAttention(nn.Module):
    def __init__(self, graph_dim, hidden_dim=64, weight_mode="static_attention"):
        super().__init__()
        if weight_mode not in ["mean", "static_attention"]:
            raise ValueError("weight_mode must be mean or static_attention")
        self.weight_mode = weight_mode
        self.alpha_mlp = None
        if self.weight_mode == "static_attention":
            self.alpha_mlp = nn.Sequential(
                nn.Linear(graph_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def _edge_logits(self, question_ids, skill_ids, difficulty_edge, z_q, z_s):
        if self.alpha_mlp is None:
            raise RuntimeError("edge logits are unavailable when weight_mode='mean'")
        q_edge = z_q.index_select(0, question_ids)
        s_edge = z_s.index_select(0, skill_ids)
        difficulty_edge = difficulty_edge.reshape(-1, 1).to(q_edge.dtype)
        alpha_input = torch.cat(
            (q_edge, s_edge, q_edge * s_edge, difficulty_edge * s_edge), dim=-1
        )
        return self.alpha_mlp(alpha_input).squeeze(-1)

    def edge_weights(self, problem_ids, multi_skill_mask, difficulty, z_q, z_s):
        valid_problem = torch.ne(problem_ids, -1)
        safe_problem = torch.where(valid_problem, problem_ids, torch.zeros_like(problem_ids)).long()
        mask = multi_skill_mask.float()
        skill_count = mask.sum(dim=-1, keepdim=True)
        valid = valid_problem & (skill_count.squeeze(-1) > 0)
        if self.weight_mode == "mean":
            alpha = mask / skill_count.clamp_min(1.0)
            return torch.where(valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))

        # Expand only the observed edges in the current batch.
        active_edge = (mask > 0.0) & valid.unsqueeze(-1)
        edge_pos = active_edge.nonzero(as_tuple=False)
        alpha = z_q.new_zeros(mask.shape)
        if edge_pos.numel() == 0:
            return alpha

        rows = edge_pos[:, 0]
        skills = edge_pos[:, 1]
        batch_difficulty = difficulty.float().reshape(problem_ids.shape[0], -1)[:, 0]
        edge_logits = self._edge_logits(
            safe_problem.index_select(0, rows),
            skills,
            batch_difficulty.index_select(0, rows),
            z_q,
            z_s,
        )

        dense_logits = z_q.new_full(mask.shape, -1e9)
        dense_logits = dense_logits.index_put((rows, skills), edge_logits)
        alpha = torch.softmax(dense_logits, dim=-1) * active_edge.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        alpha = torch.where(valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))
        return alpha

class StaticQSGNN(nn.Module):
    """Encode fixed question-skill structure and skill contribution weights."""

    def __init__(self, problem_num, concept_num, q_dim=64, hidden_dim=64,
                 layers=2, dropout=0.1, qs_adj=None, weight_mode="static_attention",
                 contrastive_neg_k=10, contrastive_tau=0.2, use_contrastive=True):
        super().__init__()
        self.problem_num = int(problem_num)
        self.concept_num = int(concept_num)
        self.problem_padding_idx = self.problem_num
        self.use_contrastive = bool(use_contrastive)
        self.contrastive_neg_k = int(contrastive_neg_k)
        self.contrastive_tau = float(contrastive_tau)
        self.problem_embedding = nn.Embedding(self.problem_num + 1, q_dim, padding_idx=self.problem_padding_idx)
        self.skill_embedding = nn.Embedding(self.concept_num, q_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.s_proj = nn.Linear(q_dim, q_dim)
        self.encoder = QSGraphEncoder(q_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.edge_attention = StaticQSEdgeAttention(
            q_dim, hidden_dim=hidden_dim, weight_mode=weight_mode
        )
        if qs_adj is None:
            qs_adj = torch.zeros(self.problem_num, self.concept_num)
        self.register_buffer("qs_adj", qs_adj.float())
        edge_index = (self.qs_adj > 0.0).nonzero(as_tuple=False)
        edge_questions = edge_index[:, 0].long()
        edge_skills = edge_index[:, 1].long()
        edge_count = torch.bincount(edge_questions, minlength=self.problem_num)
        edge_row_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.long), edge_count.cumsum(dim=0)), dim=0
        )
        self.register_buffer("edge_questions", edge_questions)
        self.register_buffer("edge_skills", edge_skills)
        self.register_buffer("edge_row_ptr", edge_row_ptr)

    def encode(self, qs_adj=None):
        adj = self.qs_adj if qs_adj is None else qs_adj.float().to(self.qs_adj.device)
        q_nodes = self.q_proj(self.problem_embedding.weight[:self.problem_num])
        s_nodes = self.s_proj(self.skill_embedding.weight)
        return self.encoder(q_nodes, s_nodes, adj)

    def sparse_edge_weights(self, problem_difficulty, z_q, z_s):
        """Compute one normalized beta value for each observed Q-S edge."""
        if self.edge_questions.numel() == 0:
            return z_q.new_zeros((0,))
        if self.edge_attention.weight_mode == "mean":
            degree = (self.edge_row_ptr[1:] - self.edge_row_ptr[:-1]).clamp_min(1)
            return degree.index_select(0, self.edge_questions).to(z_q.dtype).reciprocal()

        edge_logits = self.edge_attention._edge_logits(
            self.edge_questions,
            self.edge_skills,
            problem_difficulty.to(device=z_q.device, dtype=z_q.dtype).index_select(
                0, self.edge_questions
            ),
            z_q,
            z_s,
        )
        # Normalize sparse edge weights within each question's skill set.
        row_max = edge_logits.new_full((self.problem_num,), -torch.inf)
        row_max.scatter_reduce_(
            0, self.edge_questions, edge_logits.detach(), reduce="amax", include_self=True
        )
        edge_exp = torch.exp(edge_logits - row_max.index_select(0, self.edge_questions))
        row_sum = edge_logits.new_zeros((self.problem_num,))
        row_sum.index_add_(0, self.edge_questions, edge_exp)
        return edge_exp / row_sum.index_select(0, self.edge_questions).clamp_min(1e-8)

    def lookup_sparse_edge_weights(self, problem_ids, multi_skill_mask, edge_values):
        """Gather cached sparse beta edges into only the current batch rows."""
        valid = (problem_ids >= 0) & (multi_skill_mask.sum(dim=-1) > 0)
        safe_problem = torch.where(valid, problem_ids, torch.zeros_like(problem_ids)).long()
        beta = edge_values.new_zeros(multi_skill_mask.shape)
        valid_rows = valid.nonzero(as_tuple=False).flatten()
        if valid_rows.numel() == 0 or edge_values.numel() == 0:
            return beta

        questions = safe_problem.index_select(0, valid_rows)
        starts = self.edge_row_ptr.index_select(0, questions)
        ends = self.edge_row_ptr.index_select(0, questions + 1)
        lengths = ends - starts
        total = int(lengths.sum().item())
        if total == 0:
            return beta

        dense_rows = torch.repeat_interleave(valid_rows, lengths)
        repeated_starts = torch.repeat_interleave(starts, lengths)
        prefix = torch.cumsum(lengths, dim=0) - lengths
        offsets = torch.arange(total, device=problem_ids.device) - torch.repeat_interleave(prefix, lengths)
        edge_positions = repeated_starts + offsets
        skills = self.edge_skills.index_select(0, edge_positions)
        values = edge_values.index_select(0, edge_positions)
        selected = multi_skill_mask[dense_rows, skills] > 0.0
        beta = beta.index_put(
            (dense_rows[selected], skills[selected]), values[selected], accumulate=False
        )
        beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return beta * valid.unsqueeze(-1).to(beta.dtype)

    def contrastive_loss(self, problem_ids, multi_skill_mask, z_q, z_s):
        if not self.use_contrastive or self.contrastive_neg_k <= 0:
            return z_q.sum() * 0.0
        expanded_problem = problem_ids.unsqueeze(-1).expand_as(multi_skill_mask)
        positive = (expanded_problem != -1) & (multi_skill_mask > 0.0)
        positive_indices = positive.nonzero(as_tuple=False)
        if positive_indices.numel() == 0:
            return z_q.sum() * 0.0
        pairs = torch.stack((expanded_problem[positive].long(), positive_indices[:, -1].long()), dim=1)
        pairs = torch.unique(pairs, dim=0)
        problem_index = pairs[:, 0]
        positive_skill = pairs[:, 1]
        negative_weights = (self.qs_adj.index_select(0, problem_index) <= 0.0).float()
        has_negative = negative_weights.sum(dim=1) > 0.0
        if not has_negative.any():
            return z_q.sum() * 0.0
        problem_index = problem_index[has_negative]
        positive_skill = positive_skill[has_negative]
        negative_weights = negative_weights[has_negative]
        negative_skill = torch.multinomial(negative_weights, self.contrastive_neg_k, replacement=True)
        q_vec = F.normalize(z_q.index_select(0, problem_index), dim=-1)
        s_vec = F.normalize(z_s, dim=-1)
        positive_sim = (q_vec * s_vec.index_select(0, positive_skill)).sum(dim=-1, keepdim=True)
        negative_sim = (q_vec.unsqueeze(1) * s_vec[negative_skill]).sum(dim=-1)
        logits = torch.cat((positive_sim, negative_sim), dim=1) / max(self.contrastive_tau, 1e-8)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z_q.device)
        return F.cross_entropy(logits, labels)


class RelatedHistoryGRU(nn.Module):
    """Encode related historical interactions in chronological order."""

    def __init__(self, q_dim=64, event_dim=32, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.question_dim = int(q_dim)
        self.event_dim = int(event_dim)
        self.hidden_dim = int(hidden_dim)
        self.answer_embedding = nn.Embedding(2, self.event_dim)
        self.difficulty_mlp = nn.Sequential(nn.Linear(1, self.event_dim), nn.GELU())
        self.event_mlp = nn.Sequential(
            nn.Linear(self.question_dim + self.event_dim * 2, self.event_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.event_dim, self.event_dim),
        )
        self.history_gru = nn.GRU(self.event_dim, self.hidden_dim, batch_first=True)

    def _event_embedding(self, q_nodes, problem_ids, answers, difficulty):
        batch_size, seq_len = problem_ids.shape
        valid_event = (problem_ids >= 0) & (answers >= 0)
        safe_problem = problem_ids.clamp(min=0, max=q_nodes.shape[0] - 1).long()
        safe_answer = answers.clamp(min=0, max=1).long()
        question_event = q_nodes.index_select(0, safe_problem.reshape(-1)).view(batch_size, seq_len, self.question_dim)
        answer_event = self.answer_embedding(safe_answer)
        difficulty_event = self.difficulty_mlp(difficulty.float())
        event_emb = self.event_mlp(torch.cat((question_event, answer_event, difficulty_event), dim=-1))
        return event_emb * valid_event.unsqueeze(-1).to(event_emb.dtype)

    def forward(self, q_base, problem_ids, answers, difficulty, hist_idx, hist_len):
        batch_size, seq_len = problem_ids.shape
        history_k = hist_idx.shape[-1]
        valid_event = (problem_ids >= 0) & (answers >= 0)
        event_emb = self._event_embedding(q_base, problem_ids, answers, difficulty)

        valid_target = (hist_len > 0) & valid_event
        flat_context = event_emb.new_zeros((batch_size * seq_len, self.hidden_dim))
        target_flat = valid_target.reshape(-1).nonzero(as_tuple=False).flatten()
        if target_flat.numel() == 0:
            return flat_context.view(batch_size, seq_len, self.hidden_dim)

        target_batch = torch.div(target_flat, seq_len, rounding_mode="floor")
        target_step = torch.remainder(target_flat, seq_len)
        selected_idx = hist_idx.reshape(batch_size * seq_len, history_k).index_select(0, target_flat)
        selected_len = hist_len.reshape(-1).index_select(0, target_flat).clamp(max=history_k).long()
        slot_valid = torch.arange(history_k, device=hist_idx.device).unsqueeze(0) < selected_len.unsqueeze(1)
        safe_idx = torch.where(slot_valid, selected_idx, torch.zeros_like(selected_idx))
        safe_idx = safe_idx.clamp(min=0, max=max(seq_len - 1, 0)).long()
        if selected_idx.device.type == "cpu":
            invalid = slot_valid & ((selected_idx < 0) | (selected_idx >= target_step.unsqueeze(1)))
            if invalid.any():
                raise AssertionError("history indices must be valid and strictly earlier than target step")
        history_events = event_emb[target_batch.unsqueeze(1), safe_idx]
        history_events = history_events * slot_valid.unsqueeze(-1).to(history_events.dtype)
        packed = pack_padded_sequence(
            history_events,
            selected_len.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        flat_context = flat_context.index_copy(0, target_flat, hidden[-1])
        return flat_context.view(batch_size, seq_len, self.hidden_dim)

    def forward_step(self, q_nodes, problem_ids, answers, difficulty, hist_idx_t, hist_len_t, valid_t, target_step):
        batch_size, seq_len = problem_ids.shape
        history_k = hist_idx_t.shape[-1]
        context = q_nodes.new_zeros((batch_size, self.hidden_dim))
        valid_target = (hist_len_t > 0) & valid_t
        rows = valid_target.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            return context

        selected_idx = hist_idx_t.index_select(0, rows)
        selected_len = hist_len_t.index_select(0, rows).clamp(max=history_k).long()
        slot_valid = torch.arange(history_k, device=hist_idx_t.device).unsqueeze(0) < selected_len.unsqueeze(1)
        safe_idx = torch.where(slot_valid, selected_idx, torch.zeros_like(selected_idx))
        safe_idx = safe_idx.clamp(min=0, max=max(seq_len - 1, 0)).long()
        if selected_idx.device.type == "cpu":
            invalid = slot_valid & ((selected_idx < 0) | (selected_idx >= int(target_step)))
            if invalid.any():
                raise AssertionError("history indices must be valid and strictly earlier than target step")

        hist_problem = problem_ids[rows.unsqueeze(1), safe_idx]
        hist_answer = answers[rows.unsqueeze(1), safe_idx]
        hist_difficulty = difficulty[rows.unsqueeze(1), safe_idx]
        event_emb = self._event_embedding(q_nodes, hist_problem, hist_answer, hist_difficulty)
        event_emb = event_emb * slot_valid.unsqueeze(-1).to(event_emb.dtype)
        packed = pack_padded_sequence(
            event_emb,
            selected_len.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        return context.index_copy(0, rows, hidden[-1])


class StaticQSReadout(nn.Module):
    """Read fixed Q-S structure together with a learner's evolving skill states."""

    def __init__(self, graph_dim=64):
        super().__init__()
        self.z_w0 = nn.Linear(graph_dim, graph_dim, bias=False)
        self.z_wq = nn.Linear(graph_dim, graph_dim, bias=False)
        self.z_ws = nn.Linear(graph_dim, graph_dim, bias=False)
        self.z_wd = nn.Linear(1, graph_dim, bias=False)
        self.z_ln = nn.LayerNorm(graph_dim)

    def forward(self, q_ids, h_state, difficulty_t, q_base, s_base, beta):
        valid = q_ids >= 0
        safe_q = torch.where(valid, q_ids, torch.zeros_like(q_ids)).long()
        q_t = q_base.index_select(0, safe_q)
        sbar = torch.matmul(beta, s_base)
        hbar = (beta.unsqueeze(-1) * h_state).sum(dim=1)
        difficulty_t = difficulty_t.float().reshape(q_ids.shape[0], -1)[:, :1]
        z_q = self.z_ln(
            self.z_w0(q_t) + self.z_wq(q_t) + self.z_ws(sbar) + self.z_wd(difficulty_t)
        )
        z_q = torch.where(valid.unsqueeze(-1), z_q, torch.zeros_like(z_q))
        hbar = torch.where(valid.unsqueeze(-1), hbar, torch.zeros_like(hbar))
        return {"valid": valid, "safe_q": safe_q, "beta": beta, "hbar": hbar, "z_q": z_q}


class MPLKTModel(nn.Module):
    def __init__(
        self,
        problem_num,
        concept_num,
        combo_num,
        qs_adj,
        problem_difficulty=None,
        hid_dim=32,
        emb_dim=32,
        q_dim=64,
        fatigue_dim=16,
        fatigue_hidden_dim=64,
        history_hidden_dim=32,
        dropout=0.1,
        qs_gnn_layers=2,
        use_fatigue=True,
        skill_update_mode="combo",
        use_qs_structural_learning=True,
    ):
        super().__init__()
        if skill_update_mode not in {"combo", "first_skill"}:
            raise ValueError("skill_update_mode must be combo or first_skill")
        self.problem_num = problem_num
        self.concept_num = concept_num
        self.combo_num = combo_num
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.q_dim = q_dim
        self.use_fatigue = bool(use_fatigue)
        self.skill_update_mode = skill_update_mode
        self.use_qs_structural_learning = bool(use_qs_structural_learning)
        self.register_buffer("qs_adj", qs_adj.float())
        if problem_difficulty is None:
            problem_difficulty = torch.zeros(problem_num, dtype=torch.float32)
        problem_difficulty = torch.as_tensor(problem_difficulty, dtype=torch.float32).reshape(-1)
        if problem_difficulty.numel() != problem_num:
            raise ValueError(
                f"problem_difficulty must contain {problem_num} values, "
                f"got {problem_difficulty.numel()}"
            )
        self.register_buffer("problem_difficulty", problem_difficulty)
        graph = torch.full((concept_num, concept_num), 1.0 / max(concept_num - 1, 1), dtype=torch.float32)
        graph.fill_diagonal_(0.0)
        self.register_buffer("graph", graph)
        self.combo_response_emb = nn.Embedding(combo_num * 2, emb_dim)
        self.skill_response_emb = nn.Embedding(concept_num * 2, emb_dim)
        self.skill_emb = nn.Embedding(concept_num + 1, emb_dim, padding_idx=concept_num)
        mlp_input_dim = hid_dim + emb_dim
        self.f_self = StateUpdateMLP(mlp_input_dim, hid_dim, hid_dim, dropout=dropout)
        self.f_neighbor_in = StateUpdateMLP(2 * mlp_input_dim, hid_dim, hid_dim, dropout=dropout)
        self.f_neighbor_out = StateUpdateMLP(2 * mlp_input_dim, hid_dim, hid_dim, dropout=dropout)
        self.erase_add_gate = EraseAddGate(hid_dim, concept_num)
        self.knowledge_gru = nn.GRUCell(hid_dim, hid_dim)
        self.fatigue = VectorFatigueModule(7, fatigue_hidden_dim, fatigue_dim, dropout)
        effective_qs_layers = qs_gnn_layers if self.use_qs_structural_learning else 0
        qs_weight_mode = "static_attention" if self.use_qs_structural_learning else "mean"
        self.static_qs = StaticQSGNN(
            problem_num,
            concept_num,
            q_dim,
            q_dim,
            effective_qs_layers,
            dropout,
            qs_adj=qs_adj,
            weight_mode=qs_weight_mode,
            use_contrastive=self.use_qs_structural_learning,
        )
        self.static_qs_readout = StaticQSReadout(graph_dim=q_dim)
        self.history_gru = RelatedHistoryGRU(q_dim=q_dim, event_dim=32, hidden_dim=history_hidden_dim, dropout=dropout)
        pred_in = hid_dim + q_dim + fatigue_dim + history_hidden_dim
        self.pred_head = nn.Sequential(
            nn.Linear(pred_in, q_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(q_dim, 1),
        )
        self._epoch_q_graph = None
        self._epoch_s_graph = None
        self._epoch_beta_graph = None
        self._epoch_q_base = None
        self._epoch_s_base = None
        self._epoch_beta_values = None

    def refresh_static_cache(self, track_static_grad=None):
        """Encode Q/S nodes and sparse beta edges once for the current epoch."""
        if track_static_grad is None:
            track_static_grad = self.training and torch.is_grad_enabled()
        if track_static_grad:
            q_graph, s_graph = self.static_qs.encode()
            beta_graph = self.static_qs.sparse_edge_weights(
                self.problem_difficulty, q_graph, s_graph
            )
            self._epoch_q_graph = q_graph
            self._epoch_s_graph = s_graph
            self._epoch_beta_graph = beta_graph
            q_base = q_graph.detach().requires_grad_(True)
            s_base = s_graph.detach().requires_grad_(True)
            beta_values = beta_graph.detach()
            if beta_graph.requires_grad:
                beta_values = beta_values.requires_grad_(True)
        else:
            with torch.no_grad():
                q_base, s_base = self.static_qs.encode()
                beta_values = self.static_qs.sparse_edge_weights(
                    self.problem_difficulty, q_base, s_base
                )
            self._epoch_q_graph = None
            self._epoch_s_graph = None
            self._epoch_beta_graph = None
            q_base = q_base.detach()
            s_base = s_base.detach()
            beta_values = beta_values.detach()

        self._epoch_q_base = q_base
        self._epoch_s_base = s_base
        self._epoch_beta_values = beta_values

    def backward_static_encoder(self):
        """Transfer accumulated cache gradients through the epoch encoder graph."""
        if self._epoch_q_graph is None or self._epoch_s_graph is None:
            return False
        outputs = []
        gradients = []
        if self._epoch_q_base.grad is not None:
            outputs.append(self._epoch_q_graph)
            gradients.append(self._epoch_q_base.grad.detach())
        if self._epoch_s_base.grad is not None:
            outputs.append(self._epoch_s_graph)
            gradients.append(self._epoch_s_base.grad.detach())
        if self._epoch_beta_values.requires_grad and self._epoch_beta_values.grad is not None:
            outputs.append(self._epoch_beta_graph)
            gradients.append(self._epoch_beta_values.grad.detach())
        if outputs:
            torch.autograd.backward(outputs, grad_tensors=gradients)
        self._epoch_q_graph = None
        self._epoch_s_graph = None
        self._epoch_beta_graph = None
        return bool(outputs)

    def _first_skill_mask(self, skill_mask):
        active = skill_mask > 0.0
        out = torch.zeros_like(skill_mask)
        has_skill = active.any(dim=-1)
        pos = has_skill.nonzero(as_tuple=True)
        if len(pos) > 0 and pos[0].numel() > 0:
            first = active.float().argmax(dim=-1)
            out[(*pos, first[pos])] = 1.0
        return out

    def forward(self, batch, return_cl=False, return_load_diagnostics=False):
        problem_ids = batch["problem_ids"]
        combo_ids = batch["combo_ids"]
        correct = batch["correct"]
        mask = batch["mask"]
        skill_mask = batch["multi_skill_mask"]
        difficulty = batch["difficulty"]
        fatigue_features = batch["fatigue"]
        hist_idx = batch["hist_idx"]
        hist_len = batch["hist_len"]
        device = next(self.parameters()).device
        problem_ids = problem_ids.to(device)
        combo_ids = combo_ids.to(device)
        correct = correct.to(device)
        mask = mask.to(device)
        skill_mask = skill_mask.to(device)
        difficulty = difficulty.to(device)
        fatigue_features = fatigue_features.to(device)
        hist_idx = hist_idx.to(device)
        hist_len = hist_len.to(device)

        batch_size, seq_len = problem_ids.shape
        if self._epoch_q_base is None or self._epoch_beta_values is None:
            self.refresh_static_cache(track_static_grad=self.training and torch.is_grad_enabled())
        q_base = self._epoch_q_base
        s_base = self._epoch_s_base
        h = torch.zeros(batch_size, self.concept_num, self.hid_dim, device=device)
        preds = []
        load_diagnostics = {
            "practice_load_score": [],
            "instantaneous_load_score": [],
            "continuous_load_score": [],
        } if return_load_diagnostics else None
        model_skill_mask = skill_mask
        if self.skill_update_mode == "first_skill":
            model_skill_mask = self._first_skill_mask(skill_mask)
        for t in range(seq_len):
            valid_t = mask[:, t] & (problem_ids[:, t] >= 0)
            if self.use_fatigue:
                fatigue_vec_t, load_score_t, instantaneous_score_t, continuous_score_t = self.fatigue(
                    fatigue_features[:, t, :], mask=valid_t
                )
            else:
                fatigue_vec_t = torch.zeros(batch_size, self.fatigue.fatigue_norm.normalized_shape[0], device=device)
                load_score_t = torch.zeros(batch_size, 1, device=device)
                instantaneous_score_t = torch.zeros(batch_size, 1, device=device)
                continuous_score_t = torch.zeros(batch_size, 1, device=device)
            beta_t = self.static_qs.lookup_sparse_edge_weights(
                problem_ids[:, t],
                model_skill_mask[:, t],
                self._epoch_beta_values,
            )
            read_out = self.static_qs_readout(
                problem_ids[:, t], h, difficulty[:, t], q_base, s_base, beta_t
            )
            z_q = read_out["z_q"]
            beta = read_out["beta"]
            h_read = read_out["hbar"]
            history_ctx_t = self.history_gru.forward_step(
                q_base, problem_ids, correct, difficulty, hist_idx[:, t], hist_len[:, t], valid_t, t
            )
            pred_logit = self.pred_head(torch.cat([h_read, z_q, fatigue_vec_t, history_ctx_t], dim=-1)).squeeze(-1)
            pred_t = torch.sigmoid(pred_logit)
            pred_t = torch.where(valid_t, pred_t, torch.zeros_like(pred_t))
            preds.append(pred_t)
            h = self._update_student_h(
                h,
                combo_ids[:, t],
                correct[:, t],
                beta.detach(),
                valid_t,
            )
            if return_load_diagnostics:
                load_diagnostics["practice_load_score"].append(load_score_t.squeeze(-1))
                load_diagnostics["instantaneous_load_score"].append(instantaneous_score_t.squeeze(-1))
                load_diagnostics["continuous_load_score"].append(continuous_score_t.squeeze(-1))
        pred = torch.stack(preds, dim=1)
        cl = self.static_qs.contrastive_loss(problem_ids, model_skill_mask, q_base, s_base) if return_cl else pred.sum() * 0.0
        if return_load_diagnostics:
            diagnostics = {
                name: torch.stack(values, dim=1)[:, 1:].detach()
                for name, values in load_diagnostics.items()
            }
            return pred[:, 1:], cl, diagnostics
        return pred[:, 1:], cl

    def _build_interaction_features(self, h, combo_t, correct_t, source_alpha, valid_t):
        batch_size = h.size(0)
        device = h.device
        skill_index = torch.full((batch_size, self.concept_num), self.concept_num, device=device, dtype=torch.long)
        skill_index[valid_t] = torch.arange(self.concept_num, device=device)
        concept_embedding = self.skill_emb(skill_index)

        active = (source_alpha > 0.0) & valid_t.view(-1, 1)
        if active.any():
            rows, skills = active.nonzero(as_tuple=True)
            result_idx = correct_t[rows].clamp_min(0).long()
            if self.skill_update_mode == "first_skill":
                source_embedding = self.skill_response_emb(skills * 2 + result_idx)
            else:
                combo_idx = combo_t[rows].clamp_min(0).long()
                source_embedding = self.combo_response_emb(combo_idx * 2 + result_idx)
            concept_embedding[rows, skills] = source_embedding
        return torch.cat((h, concept_embedding), dim=-1)

    def _agg_neighbors_mixed_source(self, tmp_h, source_alpha, valid_t):
        m_next = tmp_h[:, :, :self.hid_dim].clone()
        if not valid_t.any():
            return m_next
        masked_tmp = tmp_h[valid_t]
        masked_alpha = source_alpha[valid_t].float()
        source_mix = (masked_alpha.unsqueeze(-1) * masked_tmp).sum(dim=1)
        self_features = self.f_self(source_mix)
        expanded_source = source_mix.unsqueeze(1).expand(-1, self.concept_num, -1)
        neigh_input = torch.cat((expanded_source, masked_tmp), dim=-1)
        adj = torch.matmul(masked_alpha, self.graph).unsqueeze(-1)
        reverse_adj = torch.matmul(masked_alpha, self.graph.transpose(0, 1)).unsqueeze(-1)
        neigh_features = adj * self.f_neighbor_in(neigh_input) + reverse_adj * self.f_neighbor_out(neigh_input)
        source_mask = masked_alpha > 0.0
        if source_mask.any():
            neigh_features[source_mask] = self_features.unsqueeze(1).expand(-1, self.concept_num, -1)[source_mask]
        m_next[valid_t] = neigh_features
        return m_next

    def _apply_candidate_update(self, m_next, h, valid_t):
        if not valid_t.any():
            return h.clone()
        candidate = m_next.clone()
        candidate[valid_t] = self.erase_add_gate(candidate[valid_t])
        updated = self.knowledge_gru(
            candidate[valid_t].reshape(-1, self.hid_dim),
            h[valid_t].reshape(-1, self.hid_dim),
        )
        candidate[valid_t] = updated.reshape(-1, self.concept_num, self.hid_dim)
        return candidate

    def _update_student_h(self, h, combo_t, correct_t, beta_t, valid_t):
        tmp_h = self._build_interaction_features(h, combo_t, correct_t, beta_t, valid_t)
        m_next = self._agg_neighbors_mixed_source(tmp_h, beta_t, valid_t)
        return self._apply_candidate_update(m_next, h, valid_t)
