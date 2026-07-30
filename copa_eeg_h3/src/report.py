"""Data-derived H3 evidence table and Markdown report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PATH_REPRESENTATIONS = {
    "Q": {"q"},
    "K": {"k"},
    "V": {"v"},
    "Attention": {"attention_logits", "attention_probs", "attention_output"},
    "Residual": {"residual", "residual_before_attention"},
    "Final": {"final", "pooled_representation"},
}


def evidence_table(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row for row in family_rows
        if row["head"] == "all" and row["model"] == "LogisticRegression"
    ]
    result = []
    for family in sorted({row["operator_family"] for row in rows}):
        output: dict[str, Any] = {"operator_family": family}
        path_best = {}
        for path, representations in PATH_REPRESENTATIONS.items():
            candidates = [
                row for row in rows
                if row["operator_family"] == family and row["representation"] in representations
            ]
            if candidates:
                best = max(candidates, key=lambda row: float(row["balanced_accuracy"]))
                output[path] = float(best["balanced_accuracy"])
                output[f"{path}_ci_low"] = float(best["balanced_accuracy_ci_low"])
                output[f"{path}_ci_high"] = float(best["balanced_accuracy_ci_high"])
                output[f"{path}_source"] = f"{best['representation']}@L{best['layer']}"
                path_best[path] = best
            else:
                output[path] = np.nan
        ordered = sorted(path_best, key=lambda key: float(path_best[key]["balanced_accuracy"]), reverse=True)
        if not ordered:
            output["dominant_path"] = "not_estimable"
            output["dominance_rule"] = "no valid subject-disjoint probe"
        elif len(ordered) == 1:
            output["dominant_path"] = ordered[0]
            output["dominance_rule"] = "only estimable path"
        else:
            first, second = path_best[ordered[0]], path_best[ordered[1]]
            separated = float(first["balanced_accuracy_ci_low"]) > float(second["balanced_accuracy_ci_high"])
            output["dominant_path"] = ordered[0] if separated else f"{ordered[0]} (CI overlaps)"
            output["dominance_rule"] = "highest score; dominant only if bootstrap CIs do not overlap"
        result.append(output)
    return result


def _best(rows, names):
    candidates = [
        row for row in rows
        if row["representation"] in names and row["head"] == "all"
        and row["model"] == "LogisticRegression"
    ]
    return max(candidates, key=lambda row: float(row["balanced_accuracy"])) if candidates else None


def _fmt(row):
    if row is None:
        return "NA"
    return (
        f"{float(row['balanced_accuracy']):.3f} "
        f"[{float(row['balanced_accuracy_ci_low']):.3f}, "
        f"{float(row['balanced_accuracy_ci_high']):.3f}]"
        f"（{row['representation']}，layer {row['layer']}）"
    )


def write_report(
    output_dir: Path,
    dataset: dict[str, Any],
    model: dict[str, Any],
    operator_rows: list[dict[str, Any]],
    operator_ovr_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> None:
    q, k, v = _best(operator_rows, {"q"}), _best(operator_rows, {"k"}), _best(operator_rows, {"v"})
    attention = _best(operator_rows, PATH_REPRESENTATIONS["Attention"])
    residual = _best(operator_rows, PATH_REPRESENTATIONS["Residual"])
    final = _best(operator_rows, PATH_REPRESENTATIONS["Final"])
    path_rows = {"Q": q, "K": k, "V": v, "Attention": attention, "Residual": residual, "Final": final}
    valid = {name: row for name, row in path_rows.items() if row is not None}
    chance = float(next(iter(valid.values()))["chance_level"]) if valid else np.nan
    ordered = sorted(valid, key=lambda name: float(valid[name]["balanced_accuracy"]), reverse=True)
    strongest = ordered[0] if ordered else "NA"
    qk_best = max(float(q["balanced_accuracy"]) if q else 0, float(k["balanced_accuracy"]) if k else 0)
    v_res_mlp = _best(operator_rows, {"v", "residual", "mlp_output"})
    qk_only_supported = (
        qk_best > chance + 0.05
        and v_res_mlp is not None
        and qk_best > float(v_res_mlp["balanced_accuracy_ci_high"])
    )
    def best_operator(representations):
        candidates = [
            row for row in operator_ovr_rows
            if row["representation"] in representations and row["head"] == "all"
        ]
        return max(candidates, key=lambda row: float(row["balanced_accuracy"])) if candidates else None
    q_operator, k_operator, v_operator = (
        best_operator({"q"}), best_operator({"k"}), best_operator({"v"})
    )
    if not model["pretrained"]:
        verdict = "本次为 architecture smoke test，不能据此支持或反驳神经基础模型上的 H3。"
    elif strongest in {"Q", "K", "Attention"}:
        verdict = "预训练模型结果的最强路径位于 Q/K/attention，方向上支持 H3 的 attention-affinity 版本。"
    elif strongest in {"V", "Residual", "Final"}:
        verdict = "预训练模型的最强路径不在 Q/K，当前结果反驳 Q/K-only 的强版本。"
    else:
        verdict = "可用证据不足，当前运行无法判断 H3。"
    lines = [
        "# COPA EEG H3 实验报告",
        "",
        "## 结论摘要",
        "",
        verdict,
        f"全 operator 十分类的最强 subject-disjoint 路径为 **{strongest}**。"
        if strongest != "NA" else "没有可估计的 operator probe。",
        "",
        "## 运行身份与边界",
        "",
        f"- 数据：`{dataset['dataset_name']}`，模式 `{dataset['mode']}`，"
        f"{dataset['n_subjects']} 名受试者、{dataset['n_epochs']} 个 source epochs。",
        f"- 模型：`{model['model_kind']}`；预训练：`{model['pretrained']}`；来源：{model['provenance']}。",
        f"- 参数量：{model['parameter_count']:,}；推理设备：CPU；`torch.inference_mode()`。",
        "- 所有主 probe 使用 leave-one-subject-out；StandardScaler 仅在每折训练受试者拟合。",
        "- bootstrap 单位是 held-out subject 的 fold score，而不是 view、epoch 或 token。",
        "",
        "## 与 H1/H2 当前结果的衔接",
        "",
        "- H1 的 9-subject、720-epoch BNCI2014_001 结果支持 operator 跨受试者可识别："
        "最佳 balanced accuracy=0.7952（chance=0.0909），cross-operator task drop=0.0724。",
        "- H2 的 5-subject、40-epoch 结果仅部分支持低秩可迁移结构：平均 ρ(8)=0.637，"
        "LOSO train-subspace 相对随机子空间优势=0.516，operator-specificity gap=0.144；"
        "但 rank 8 没有比较通过 Holm 校正。",
        "- H3 因而不再重复证明“有没有 operator 信息”，而是定位这些信息穿过 Transformer 的具体路径。",
        "",
        "## 总体路径证据",
        "",
        "| Path | Best balanced accuracy (subject bootstrap CI) |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {_fmt(row)} |" for name, row in path_rows.items())
    lines.extend([
        "",
        "## Operator-family evidence table",
        "",
        "| Operator family | Q | K | V | Attention | Residual | Final | Dominant path |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in evidence:
        lines.append(
            f"| {row['operator_family']} | "
            + " | ".join(f"{float(row[path]):.3f}" for path in ("Q", "K", "V", "Attention", "Residual", "Final"))
            + f" | {row['dominant_path']} |"
        )
    lines.extend([
        "",
        "Dominant path 先取最高 balanced accuracy；只有其 bootstrap CI 下界高于第二名上界时才视为明确主导，"
        "否则标记 `CI overlaps`。chance 增量可由 evidence CSV 的 chance level 复核。",
        "",
        "## 对 H3 七个问题的回答",
        "",
        f"1. **哪些 operator 在 Q/K 中最可分？** Q 最强为 "
        f"`{q_operator['operator_label'] if q_operator else 'NA'}`（{_fmt(q_operator)}）；"
        f"K 最强为 `{k_operator['operator_label'] if k_operator else 'NA'}`（{_fmt(k_operator)}）。"
        "逐 family 结果见上表。",
        f"2. **哪些 operator 在 V 中最可分？** "
        f"`{v_operator['operator_label'] if v_operator else 'NA'}`（{_fmt(v_operator)}）；V 没有被选择性排除。",
        f"3. **final representation 是否仍保留大量 operator 信息？** 最佳 final={_fmt(final)}；"
        "应相对 chance 和 Q/K/V 的 CI 同时解释。",
        f"4. **Q/K-only projection 是否有实证依据？** {'当前 CI 分离规则下有方向性依据。' if qk_only_supported else '当前 CI 分离规则下没有充分依据。'}",
        f"5. **是否需要处理 V、residual 或 MLP？** 最强 V/residual/MLP={_fmt(v_res_mlp)}；"
        "若接近或高于 Q/K，就不能只处理 Q/K。",
        f"6. **支持、部分支持还是反驳 H3？** {verdict}",
        f"7. **pretrained 还是 fallback？** `pretrained={model['pretrained']}`；{model['provenance']}。",
        "",
        "## Attention 与负对照",
        "",
        "- `attention_statistics.csv` 含逐 source/operator/layer/head 的 entropy、diagonal/off-diagonal mass、top-k concentration。",
        "- `attention_distance.csv` 对同一 source epoch 的 identity/operator attention map 计算 Frobenius 与 Jensen-Shannon 距离。",
        "- `controls.csv` 含 shuffled operator labels、shuffled subject split labels、random-initialized model、"
        "identity duplicate、dataset/source ID、same-operator/different-source 与 same-source/different-operator。",
        "",
        "## 解释限制",
        "",
        "- fallback 模型只验证张量提取、统计和报告链路，不能冒充 CBraMod 或其他 pretrained EEG foundation model 结论。",
        "- 单数据集时 dataset-ID classifier 不可估计，程序明确记录为 `not_estimable_single_dataset`。",
        "- 多分类 probe 反映可识别信息，不单独证明因果污染路径。",
        "- H1/H2 数值来自仓库中现有真实运行报告，用作任务承接背景；H3 的新判定只使用本次 H3 输出。",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
