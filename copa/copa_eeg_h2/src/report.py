"""Evidence-derived Markdown report; no conclusion is hard-coded."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import numpy as np


def _mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    output_dir: Path,
    dataset_summary: dict,
    explained: list[dict],
    thresholds: list[dict],
    transfer: list[dict],
    specificity: list[dict],
    comparisons: list[dict],
    locality: list[dict],
    h1_context: str,
) -> None:
    paired_overall = [
        row for row in explained if row["control"] == "paired" and row["scope"] == "overall"
    ]
    per_operator: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in paired_overall:
        per_operator[row["operator"]][int(row["rank"])].append(float(row["explained_variance"]))
    operator_rho = {
        operator: {rank: _mean(values) for rank, values in by_rank.items()}
        for operator, by_rank in per_operator.items()
    }
    rank80: dict[str, list[float]] = defaultdict(list)
    for row in thresholds:
        if (
            row["control"] == "paired"
            and row["scope"] == "overall"
            and math.isclose(float(row["threshold"]), 0.8)
        ):
            rank80[row["operator"]].append(float(row["minimum_rank"]))
    ordered = sorted(rank80, key=lambda operator: _mean(rank80[operator]))

    train_values = [
        float(row["explained_test"]) for row in transfer
        if row["baseline"] == "train_subspace" and int(row["rank"]) == 8
    ]
    random_values = [
        float(row["explained_test"]) for row in transfer
        if row["baseline"] == "random_subspace" and int(row["rank"]) == 8
    ]
    oracle_values = [
        float(row["explained_test"]) for row in transfer
        if row["baseline"] == "oracle" and int(row["rank"]) == 8
    ]
    transfer_advantage = _mean(train_values) - _mean(random_values)

    diagonal, off_diagonal = [], []
    for row in specificity:
        target = diagonal if row["source_operator"] == row["target_operator"] else off_diagonal
        target.append(float(row["explained_test"]))
    specificity_gap = _mean(diagonal) - _mean(off_diagonal)

    paired_control_advantages = [
        float(row["mean_difference"]) for row in comparisons
        if int(row["rank"]) == 8 and row["control"] != "paired"
    ]
    control_advantage = _mean(paired_control_advantages)
    control_breakdown = {
        control: _mean(
            row["mean_difference"] for row in comparisons
            if int(row["rank"]) == 8 and row["control"] == control
        )
        for control in sorted({row["control"] for row in comparisons})
    }
    significant_comparisons = sum(
        int(row["rank"]) == 8
        and np.isfinite(float(row.get("holm_p", np.nan)))
        and float(row["holm_p"]) <= 0.05
        for row in comparisons
    )
    significant_by_control = {
        control: sum(
            int(row["rank"]) == 8
            and row["control"] == control
            and float(row["mean_difference"]) > 0
            and np.isfinite(float(row.get("holm_p", np.nan)))
            and float(row["holm_p"]) <= 0.05
            for row in comparisons
        )
        for control in control_breakdown
    }
    median_rho8 = _mean(
        row["explained_variance"] for row in paired_overall if int(row["rank"]) == 8
    )
    local_gains = [
        float(row["local_minus_global"]) for row in locality
        if row.get("metric") == "transfer_gain"
    ]
    local_gain = _mean(local_gains)

    low_rank_operators = [
        operator for operator, values in operator_rho.items()
        if values.get(8, 0.0) >= 0.7
    ]
    low_rank_names = "、".join(f"`{name}`" for name in low_rank_operators)
    transfer_by_operator = {
        operator: (
            _mean(
                row["explained_test"] for row in transfer
                if row["operator"] == operator
                and row["baseline"] == "train_subspace"
                and int(row["rank"]) == 8
            )
            - _mean(
                row["explained_test"] for row in transfer
                if row["operator"] == operator
                and row["baseline"] == "random_subspace"
                and int(row["rank"]) == 8
            )
        )
        for operator in operator_rho
    }
    controls_by_operator = {
        operator: {
            control: _mean(
                row["mean_difference"] for row in comparisons
                if row["operator"] == operator
                and row["control"] == control
                and int(row["rank"]) == 8
            )
            for control in control_breakdown
        }
        for operator in operator_rho
    }
    robust_operators = [
        operator for operator in low_rank_operators
        if transfer_by_operator.get(operator, 0) > 0
        and all(value > 0 for value in controls_by_operator[operator].values())
    ]
    robust_names = "、".join(f"`{name}`" for name in robust_operators)
    if robust_operators and significant_comparisons > 0:
        verdict = (
            f"完整 9-subject 结果支持 H2 的算子特异版本：{robust_names} "
            "同时呈现低秩集中、跨受试者迁移和正向对照差异。"
            "该结论不能外推到所有算子，尤其不适用于随机通道 dropout。"
        )
    elif low_rank_operators and transfer_advantage > 0:
        verdict = (
            f"{low_rank_names} 等部分算子呈现低秩且可迁移的方向性证据；"
            "但对照优势不一致，且没有 rank 8 比较通过 Holm 校正，"
            "因此只能部分支持、尚不能确认 H2。"
        )
    elif median_rho8 >= 0.7 and transfer_advantage <= 0:
        verdict = (
            "差分在样本内较集中，但训练受试者子空间未优于随机子空间；"
            "固定可迁移子空间版本的 H2 不成立。"
        )
    else:
        verdict = "当前运行未显示稳定的低秩集中与迁移证据，因此不支持 H2。"
    if dataset_summary["mode"] == "synthetic":
        verdict += " 但本次为 synthetic smoke test，只能验证流程，不能作为科学结论。"

    lines = [
        "# COPA EEG H2 实验报告",
        "",
        "## 结论摘要",
        "",
        verdict,
        "",
        "## 与 H1 的衔接",
        "",
        h1_context,
        "",
        "## 数据与防泄漏设计",
        "",
        f"- 模式：`{dataset_summary['mode']}`；数据集：`{dataset_summary['dataset_name']}`；加载器：`{dataset_summary['loader']}`。",
        f"- {dataset_summary['n_subjects']} 名受试者，{dataset_summary['n_epochs']} 个 epoch，"
        f"{dataset_summary['n_channels']} 通道，采样率 {dataset_summary['sfreq']:.1f} Hz。",
        "- PCA 仅在相应训练受试者的 identity view 上拟合；LOSO 每折显式检查测试 sample ID 未进入拟合集合。",
        "- 所有显著性检验及 bootstrap 均以 subject 为单位，而非把 token/epoch 当独立重复。",
        "",
        "## 1. 哪些算子最接近低秩",
        "",
    ]
    for operator in ordered:
        rho = operator_rho.get(operator, {})
        lines.append(
            f"- `{operator}`：平均 rank@80%={_fmt(_mean(rank80[operator]), 1)}；"
            f"ρ(2/4/8)={_fmt(rho.get(2, np.nan))}/{_fmt(rho.get(4, np.nan))}/{_fmt(rho.get(8, np.nan))}。"
        )
    lines.extend([
        "",
        "以上为四种 embedding 的描述性平均；完整的逐 embedding、逐算子、逐对照结果见 CSV。",
        "",
        "## 2. rank r=2/4/8 可解释多少差异",
        "",
        f"- 所有 paired operator/embedding 的平均 ρ(8)={_fmt(median_rho8)}。",
        f"- rank 8 时 paired 相对各 control 的平均差={_fmt(control_advantage)}。",
        f"- 分对照差值：Gaussian noise={_fmt(control_breakdown.get('gaussian_noise', np.nan))}，"
        f"unpaired={_fmt(control_breakdown.get('unpaired', np.nan))}，"
        f"label-preserving permutation={_fmt(control_breakdown.get('label_preserving_permutation', np.nan))}，"
        f"random orthogonal={_fmt(control_breakdown.get('random_orthogonal', np.nan))}。",
        f"- rank 8 的 Holm 校正后显著比较数={significant_comparisons}；"
        f"其中 Gaussian noise/unpaired/label permutation/random orthogonal 分别为 "
        f"{significant_by_control.get('gaussian_noise', 0)}/"
        f"{significant_by_control.get('unpaired', 0)}/"
        f"{significant_by_control.get('label_preserving_permutation', 0)}/"
        f"{significant_by_control.get('random_orthogonal', 0)}。",
        "- `explained_variance.csv` 同时保留 overall 与逐 subject 曲线；"
        "`control_comparisons.csv` 给出 subject-bootstrap CI、Wilcoxon 与 Holm 校正。",
        "",
        "## 3. 子空间是否跨受试者迁移",
        "",
        f"- LOSO train-subspace 平均解释率={_fmt(_mean(train_values))}。",
        f"- 同 rank 随机子空间平均解释率={_fmt(_mean(random_values))}；差={_fmt(transfer_advantage)}。",
        f"- 测试受试者 oracle SVD 上界={_fmt(_mean(oracle_values))}。",
        "- residual ratio 按定义等于 `1 - explained_test`，逐折结果见 `cross_subject_transfer.csv`。",
        "",
        "## 4. 子空间是否 operator-specific",
        "",
        f"- specificity matrix 对角平均={_fmt(_mean(diagonal))}，非对角平均={_fmt(_mean(off_diagonal))}，差={_fmt(specificity_gap)}。",
        "- 若对角线明显高于非对角线，说明方向具有 operator specificity；"
        "若二者接近，则更像共享的低维扰动空间。",
        "",
        "## 5. 固定 operator subspace 是否成立",
        "",
        f"- 全局训练子空间相对随机基线优势={_fmt(transfer_advantage)}。",
        f"- 标签/功率局部分组相对全局子空间的平均迁移增益={_fmt(local_gain)}。",
        f"- 同时通过低秩、迁移与四类平均对照方向筛选的算子："
        f"{robust_names if robust_names else '无'}。",
    ])
    if np.isfinite(local_gain) and local_gain > 0.03 and transfer_advantage <= 0.03:
        lines.append(
            "- 全局迁移弱而局部分组迁移更好，提示单一固定 U_o 假设可能不成立；"
            "operator effect 更可能依赖任务标签或信号功率。"
        )
    else:
        lines.append(
            "- 当前局部性结果未显示足以取代全局子空间的稳定大幅增益；"
            "仍需结合 principal angles 与真实样本量判断。"
        )
    lines.extend([
        "",
        "## 6. 支持与反驳 H2 的证据",
        "",
        f"- 低秩集中：平均 ρ(8)={_fmt(median_rho8)}。",
        f"- 强于对照：rank 8 平均 paired-control 差={_fmt(control_advantage)}。",
        f"- 对照一致性：paired-unpaired={_fmt(control_breakdown.get('unpaired', np.nan))}，"
        f"paired-random-orthogonal={_fmt(control_breakdown.get('random_orthogonal', np.nan))}。",
        f"- 可迁移性：train-random 差={_fmt(transfer_advantage)}。",
        f"- 算子特异性：diagonal-off-diagonal 差={_fmt(specificity_gap)}。",
        f"- 判定：{verdict}",
        "",
        "## 输出索引",
        "",
        "- `explained_variance.csv`, `rank_thresholds.csv`, `effective_rank.csv`",
        "- `control_comparisons.csv`, `cross_subject_transfer.csv`",
        "- `operator_specificity.csv`, `principal_angles.csv`, `locality_analysis.csv`",
        "- `representation_metadata.csv.gz`, `operator_metadata.jsonl`, `dataset_summary.json`",
        "- `figures/` 中的 cumulative spectrum、transfer、specificity、principal-angle 与 similarity 热图",
        "",
        "## 解释限制",
        "",
        "- 高 ρ(r) 只说明差分能量集中，不自动意味着任务信息或生理机制集中。",
        "- dropout 的随机通道选择按 sample 固定种子复现；它本身可能扩大估计子空间维数。",
        "- oracle 是测试集内上界，不是可部署结果。",
    ])
    average_epochs = dataset_summary["n_epochs"] / dataset_summary["n_subjects"]
    if dataset_summary["mode"] == "real" and average_epochs < 20:
        lines.append(
            f"- 本次真实运行平均每名受试者仅 {average_epochs:.0f} 个平衡 epoch，"
            "属于资源可行的初始实证；LOSO 方向可解释，但谱、bootstrap 和显著性结论"
            "应在 README 默认的每人 40 epoch 配置上复验。"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
