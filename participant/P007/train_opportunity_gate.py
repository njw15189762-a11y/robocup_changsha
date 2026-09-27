"""H010：固定反事实数据上的局部机会识别，比较 32×32 与 64×64。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


class OpportunityNet(nn.Module):
    """输入只含局部合法信息，分别预测八个残差方向的正收益概率。"""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """不依赖额外库的 average precision；无正例时返回 NaN。"""
    labels = labels.astype(bool)
    positives = int(np.sum(labels))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(np.sum(precision[sorted_labels]) / positives)


def _report(probabilities: np.ndarray, deltas: np.ndarray,
            episode_ids: np.ndarray) -> dict:
    state_scores = np.max(probabilities, axis=1)
    state_labels = np.max(deltas, axis=1) > 1e-12
    chosen = np.argmax(probabilities, axis=1)
    chosen_deltas = deltas[np.arange(len(deltas)), chosen]
    count = max(1, math.ceil(0.01 * len(deltas)))
    top = np.argsort(-state_scores, kind="stable")[:count]

    episode_best = []
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        selected = indices[np.argmax(state_scores[indices])]
        episode_best.append((state_scores[selected], chosen_deltas[selected]))
    episode_best.sort(key=lambda item: item[0], reverse=True)
    episode_count = max(1, math.ceil(0.1 * len(episode_best)))
    top_episodes = episode_best[:episode_count]

    return {
        "states": int(len(deltas)),
        "positive_states": int(np.sum(state_labels)),
        "state_positive_rate": float(np.mean(state_labels)),
        "state_average_precision": _average_precision(state_scores, state_labels),
        "action_average_precision": _average_precision(
            probabilities.reshape(-1), (deltas > 1e-12).reshape(-1)
        ),
        "top_1pct_state_count": int(count),
        "top_1pct_has_any_helpful_action": int(np.sum(state_labels[top])),
        "top_1pct_chosen_helpful": int(np.sum(chosen_deltas[top] > 1e-12)),
        "top_1pct_chosen_harmful": int(np.sum(chosen_deltas[top] < -1e-12)),
        "top_1pct_chosen_delta_j_sum": float(np.sum(chosen_deltas[top])),
        "top_10pct_episode_count": int(episode_count),
        "top_10pct_episode_chosen_delta_j_sum": float(
            sum(delta for _, delta in top_episodes)
        ),
    }


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["features"], dtype=np.float32),
            np.asarray(data["deltas"], dtype=np.float32),
            np.asarray(data["episode_ids"], dtype=np.int32),
        )


def train_one(train_x: np.ndarray, train_y: np.ndarray, train_episodes: np.ndarray,
              val_x: np.ndarray, val_y: np.ndarray, val_episodes: np.ndarray,
              *, hidden: int, seed: int, epochs: int, out: Path) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = OpportunityNet(train_x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((8,), 8.0))
    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy((train_y > 1e-12).astype(np.float32))
    x_val = torch.from_numpy(val_x)
    # 适度重采样正例状态；同一个种子使两种网络宽度看到相同索引序列。
    positive_states = np.max(train_y, axis=1) > 1e-12
    sampling_weights = np.where(positive_states, 10.0, 1.0).astype(np.float64)
    sampling_weights /= np.sum(sampling_weights)
    epoch_losses = []
    for _ in range(epochs):
        selected = rng.choice(len(train_x), size=len(train_x), replace=True,
                              p=sampling_weights)
        losses = []
        model.train()
        for start in range(0, len(selected), 256):
            batch = selected[start:start + 256]
            logits = model(x_train[batch])
            loss = criterion(logits, y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x_train)).numpy()
        val_prob = torch.sigmoid(model(x_val)).numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    return {
        "hidden": hidden,
        "model_seed": seed,
        "epochs": epochs,
        "final_training_loss": epoch_losses[-1],
        "train": _report(train_prob, train_y, train_episodes),
        "validation": _report(val_prob, val_y, val_episodes),
        "model_path": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H010 局部机会识别网络宽度对照")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"结果已存在，避免覆盖：{args.out}")
    raw_train_x, train_y, train_episodes = _load(args.train)
    raw_val_x, val_y, val_episodes = _load(args.validation)
    if raw_train_x.shape[1] != raw_val_x.shape[1]:
        raise ValueError("训练与验证特征维数不一致")
    mean = np.mean(raw_train_x, axis=0)
    std = np.std(raw_train_x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = np.clip((raw_train_x - mean) / std, -5.0, 5.0).astype(np.float32)
    val_x = np.clip((raw_val_x - mean) / std, -5.0, 5.0).astype(np.float32)
    torch.set_num_threads(1)
    results = []
    for seed in (7101, 7102, 7103):
        for hidden in (32, 64):
            model_path = args.out.with_name(
                f"{args.out.stem}-width{hidden}-seed{seed}.pt"
            )
            result = train_one(
                train_x, train_y, train_episodes,
                val_x, val_y, val_episodes,
                hidden=hidden, seed=seed, epochs=args.epochs, out=model_path,
            )
            results.append(result)
            print(
                f"seed={seed} width={hidden} "
                f"train_AP={result['train']['state_average_precision']:.4f} "
                f"val_AP={result['validation']['state_average_precision']:.4f} "
                f"val_top1_helpful={result['validation']['top_1pct_chosen_helpful']}/"
                f"{result['validation']['top_1pct_state_count']}",
                flush=True,
            )
    summary = {
        "experiment": "H010-opportunity-gate-width",
        "train_dataset": str(args.train),
        "validation_dataset": str(args.validation),
        "feature_dim": int(train_x.shape[1]),
        "train_episode_count": int(len(np.unique(train_episodes))),
        "validation_episode_count": int(len(np.unique(val_episodes))),
        "normalization": "训练集均值标准差，零方差置一，裁剪到 [-5,5]",
        "sampling": "正例状态权重 10，正动作 BCE 权重 8",
        "warning": "本实验只测规则轨迹上的离线可识别性，未接入实时策略。",
        "results": results,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"详细结果：{args.out}")


if __name__ == "__main__":
    main()
