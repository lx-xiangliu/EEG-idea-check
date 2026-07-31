"""Shared experiment orchestration, caching, serialization, and reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import platform
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from .data import EEGDataset, load_dataset
from .evaluation import evaluate_cross_operator, evaluate_operator_probe
from .features import FeatureExtractor
from .operators import EEGOperator, build_operators
from .plotting import plot_confusion_matrix, plot_heatmap
from .statistics import paired_cross_operator_test, per_operator_tests


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _dataset_signature(
    dataset: EEGDataset,
    feature_config: dict[str, Any],
    operators: list[EEGOperator],
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.X.shape).encode())
    digest.update(dataset.X.tobytes())
    digest.update(dataset.y.tobytes())
    digest.update(dataset.subjects.tobytes())
    digest.update(yaml.safe_dump(feature_config, sort_keys=True).encode())
    operator_spec = [
        (type(operator).__name__, operator.name, repr(vars(operator)))
        for operator in operators
    ]
    digest.update(repr(operator_spec).encode())
    digest.update(str(seed).encode())
    return digest.hexdigest()[:16]


def prepare_operator_features(
    dataset: EEGDataset,
    operators: list[EEGOperator],
    feature_config: dict[str, Any],
    cache_root: Path,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, Any]]:
    """Apply operators and cache traditional features per operator."""

    extractor = FeatureExtractor(feature_config)
    signature = _dataset_signature(dataset, feature_config, operators, seed)
    cache_dir = cache_root / signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_by_operator: dict[str, np.ndarray] = {}
    shared_schema: tuple[str, ...] | None = None
    cache_status: dict[str, str] = {}
    metadata_path = output_dir / "operator_metadata.jsonl"
    metadata_records: list[dict[str, Any]] = []

    for operator_index, operator in enumerate(operators):
        feature_path = cache_dir / f"{operator.name}.npz"
        if feature_path.exists():
            cached = np.load(feature_path, allow_pickle=False)
            features = cached["features"]
            schema = tuple(cached["schema"].astype(str).tolist())
            cache_status[operator.name] = "hit"
        else:
            transformed: list[np.ndarray] = []
            for epoch_index, epoch in enumerate(dataset.X):
                epoch_seed = seed + operator_index * 1_000_003 + epoch_index
                result = operator.transform(
                    epoch,
                    dataset.sfreq,
                    dataset.channel_names,
                    np.random.default_rng(epoch_seed),
                )
                transformed.append(result.data)
                record = {
                    "cache_signature": signature,
                    "epoch_index": epoch_index,
                    "subject": int(dataset.subjects[epoch_index]),
                    "task_label": int(dataset.y[epoch_index]),
                    **result.metadata,
                }
                metadata_records.append(record)
            feature_set = extractor.transform(
                np.stack(transformed), dataset.sfreq, dataset.channel_names
            )
            features, schema = feature_set.values, feature_set.schema
            np.savez_compressed(
                feature_path,
                features=features,
                schema=np.asarray(schema, dtype=str),
            )
            cache_status[operator.name] = "miss"
        if features.shape[0] != dataset.X.shape[0]:
            raise RuntimeError(f"Cached feature row mismatch for {operator.name}")
        if not np.isfinite(features).all():
            raise ValueError(f"Cached features for {operator.name} contain NaN/Inf")
        if shared_schema is None:
            shared_schema = schema
        elif schema != shared_schema:
            raise RuntimeError(f"Feature schema mismatch for {operator.name}")
        features_by_operator[operator.name] = features

    # Metadata is generated on a cold cache. Preserve an existing complete file
    # on cache hits; otherwise regenerate metadata without retaining raw views.
    expected_records = dataset.X.shape[0] * len(operators)
    existing_lines = 0
    existing_signature: str | None = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            if first_line:
                existing_lines = 1 + sum(1 for _ in handle)
                try:
                    existing_signature = json.loads(first_line).get("cache_signature")
                except json.JSONDecodeError:
                    existing_signature = None
    metadata_is_current = (
        existing_lines == expected_records and existing_signature == signature
    )
    if len(metadata_records) != expected_records and not metadata_is_current:
        metadata_records = []
        for operator_index, operator in enumerate(operators):
            for epoch_index, epoch in enumerate(dataset.X):
                result = operator.transform(
                    epoch,
                    dataset.sfreq,
                    dataset.channel_names,
                    np.random.default_rng(
                        seed + operator_index * 1_000_003 + epoch_index
                    ),
                )
                metadata_records.append(
                    {
                        "cache_signature": signature,
                        "epoch_index": epoch_index,
                        "subject": int(dataset.subjects[epoch_index]),
                        "task_label": int(dataset.y[epoch_index]),
                        **result.metadata,
                    }
                )
    if metadata_records:
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in metadata_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(
        output_dir / "feature_schema.json",
        {
            "feature_count": len(shared_schema or ()),
            "features": list(shared_schema or ()),
            "enabled_config": feature_config,
        },
    )
    return features_by_operator, shared_schema or (), {
        "signature": signature,
        "directory": str(cache_dir),
        "status": cache_status,
    }


def _markdown_matrix(matrix: np.ndarray, names: list[str]) -> str:
    header = "| train \\ test | " + " | ".join(names) + " |"
    separator = "|" + "---|" * (len(names) + 1)
    rows = [
        "| "
        + name
        + " | "
        + " | ".join(f"{value:.4f}" for value in matrix[index])
        + " |"
        for index, name in enumerate(names)
    ]
    return "\n".join([header, separator, *rows])


def save_operator_results(result: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "operator_probe_results.json", result)
    names = result["operator_names"]
    for model_name, metrics in result["models"].items():
        matrix = np.asarray(metrics["confusion_matrix"])
        path = output_dir / f"operator_confusion_matrix_{model_name}.png"
        plot_confusion_matrix(matrix, names, path, f"Operator probe: {model_name}")
    preferred = "logistic_regression"
    if preferred not in result["models"]:
        preferred = next(iter(result["models"]))
    generic = output_dir / "operator_confusion_matrix.png"
    shutil.copyfile(
        output_dir / f"operator_confusion_matrix_{preferred}.png",
        generic,
    )


def save_cross_results(result: dict[str, Any], output_dir: Path) -> None:
    write_json(output_dir / "cross_operator_results.json", result)
    names = result["operator_names"]
    matrix = np.asarray(result["balanced_accuracy_matrix"])
    frame = pd.DataFrame(matrix, index=names, columns=names)
    frame.index.name = "train_operator"
    frame.to_csv(output_dir / "cross_operator_balanced_accuracy.csv")
    markdown = "# Cross-operator balanced accuracy\n\n" + _markdown_matrix(matrix, names) + "\n"
    (output_dir / "cross_operator_balanced_accuracy.md").write_text(
        markdown, encoding="utf-8"
    )
    plot_heatmap(
        matrix,
        names,
        output_dir / "cross_operator_heatmap.png",
        output_dir / "cross_operator_heatmap.pdf",
        "Cross-operator task generalization",
    )


def _runtime_metadata(start_time: float) -> dict[str, Any]:
    elapsed = time.perf_counter() - start_time
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux commonly reports KiB.
    peak_mb = peak / (1024**2) if sys.platform == "darwin" else peak / 1024
    return {
        "elapsed_seconds": elapsed,
        "peak_memory_mb": float(peak_mb),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "cpu_only": True,
    }


def write_report(
    output_dir: Path,
    dataset: EEGDataset,
    operator_result: dict[str, Any],
    cross_result: dict[str, Any],
    statistics: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    model_scores = {
        name: result["balanced_accuracy"]
        for name, result in operator_result["models"].items()
    }
    best_model = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]
    chance = operator_result["chance_balanced_accuracy"]
    drop = cross_result["average_cross_operator_drop"]
    largest_outgoing = max(
        cross_result["outgoing_drop"], key=cross_result["outgoing_drop"].get
    )
    largest_incoming = max(
        cross_result["incoming_drop"], key=cross_result["incoming_drop"].get
    )
    if dataset.mode == "synthetic":
        conclusion = (
            "本次为 synthetic smoke-test，仅验证流程和输出契约；结果不能支持或反驳 H1，"
            "不得作为 EEG 科学结论。"
        )
        mode_warning = (
            "> **重要：synthetic flow-check 不是正式实验。以下数值只能说明代码路径可运行，"
            "不能解释为真实 EEG 证据。**"
        )
    else:
        supports = best_score > chance and drop > 0
        test_status = statistics["paired_test"].get("status")
        p_value = statistics["paired_test"].get("p_value")
        if supports and test_status == "ok" and p_value is not None and p_value < 0.05:
            conclusion = "在当前真实数据及预注册判据下，结果支持 H1。"
        elif supports:
            conclusion = (
                "当前真实数据的效应方向与 H1 一致，但统计证据尚不足，"
                "不能据此确认 H1。"
            )
        else:
            conclusion = "当前真实数据不足以支持 H1。"
        mode_warning = "本报告来自 real-data mode。"
    paired_test = statistics["paired_test"]
    if paired_test.get("p_value") is not None:
        significance = f"Wilcoxon 单侧检验 p={paired_test['p_value']:.4g}。"
    elif paired_test.get("status") == "insufficient_sample_size":
        significance = (
            f"仅有 {paired_test['n_subjects']} 名受试者，少于预设的 "
            f"{paired_test['minimum_subjects']} 人，因此不报告显著性。"
        )
    else:
        significance = paired_test.get("message", "统计检验不可用。")
    report = f"""# COPA EEG H1 实验报告

{mode_warning}

## 数据与验证设计

- 数据模式：`{dataset.mode}`
- 数据集：`{dataset.dataset_name}`
- 受试者数：{np.unique(dataset.subjects).size}
- epoch 数：{len(dataset.y)}
- 通道数：{dataset.X.shape[1]}
- 所有模型评估均按 subject ID 分组；训练受试者与测试受试者不重叠。
- scaler 位于 sklearn Pipeline 内，仅在每个训练折拟合。

## Operator separability

最佳模型为 `{best_model}`，跨受试者 balanced accuracy 为 {best_score:.4f}，
随机机会水平为 {chance:.4f}。在当前运行中，观测算子
{"呈现跨受试者可识别性" if best_score > chance else "未呈现高于机会水平的跨受试者可识别性"}。

各模型 balanced accuracy：

{chr(10).join(f"- `{name}`: {score:.4f}" for name, score in model_scores.items())}

## Cross-operator task generalization

- diagonal mean：{cross_result['diagonal_mean']:.4f}
- off-diagonal mean：{cross_result['off_diagonal_mean']:.4f}
- average cross-operator drop：{drop:.4f}
- 最大 outgoing drop：`{largest_outgoing}`（{cross_result['outgoing_drop'][largest_outgoing]:.4f}）
- 最大 incoming drop：`{largest_incoming}`（{cross_result['incoming_drop'][largest_incoming]:.4f}）

当前运行中，cross-operator drop {"存在正向数值差" if drop > 0 else "未呈现正向数值差"}。
完整矩阵与每个 cell 的 subject-bootstrap 95% CI 见相邻 CSV、Markdown 和 JSON。

## 统计检验

配对单位为受试者，默认检验为 Wilcoxon signed-rank；多算子比较使用 Holm 校正。
{significance}

## 对 H1 的判断

{conclusion}

## 资源记录

- 运行时间：{runtime['elapsed_seconds']:.2f} 秒
- 峰值内存：{runtime['peak_memory_mb']:.2f} MiB
- CPU-only：{runtime['cpu_only']}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_experiment(
    config: dict[str, Any],
    output_dir: Path,
    *,
    run_operator: bool = True,
    run_cross: bool = True,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    seed = int(config["project"]["seed"])
    n_jobs = int(config["project"].get("n_jobs", 1))
    cache_root_value = Path(config["project"]["cache_dir"])
    if not cache_root_value.is_absolute():
        cache_root_value = Path(__file__).resolve().parents[1] / cache_root_value
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    dataset = load_dataset(config, seed, cache_root_value / "data")
    dataset.validate()
    operators = build_operators(config["operators"])
    features, schema, cache_info = prepare_operator_features(
        dataset,
        operators,
        config["features"],
        cache_root_value / "features",
        output_dir,
        seed,
    )
    write_json(
        output_dir / "dataset_summary.json",
        {
            "mode": dataset.mode,
            "dataset_name": dataset.dataset_name,
            "shape": list(dataset.X.shape),
            "subjects": np.unique(dataset.subjects).astype(int).tolist(),
            "class_counts": {
                str(int(label)): int((dataset.y == label).sum())
                for label in np.unique(dataset.y)
            },
            "sfreq": dataset.sfreq,
            "channel_names": list(dataset.channel_names),
            "cache": cache_info,
        },
    )
    results: dict[str, Any] = {
        "mode": dataset.mode,
        "feature_count": len(schema),
        "cache": cache_info,
    }
    operator_result: dict[str, Any] | None = None
    cross_result: dict[str, Any] | None = None
    if run_operator:
        evaluation = config["evaluation"]
        operator_result = evaluate_operator_probe(
            features,
            dataset.subjects,
            list(evaluation["operator_models"]),
            int(evaluation["n_splits"]),
            int(evaluation["bootstrap_iterations"]),
            float(evaluation["confidence_level"]),
            seed,
            n_jobs,
        )
        save_operator_results(operator_result, output_dir)
        results["operator_probe"] = operator_result
    if run_cross:
        evaluation = config["evaluation"]
        cross_result = evaluate_cross_operator(
            features,
            dataset.y,
            dataset.subjects,
            str(evaluation["task_model"]),
            int(evaluation["bootstrap_iterations"]),
            float(evaluation["confidence_level"]),
            seed,
            n_jobs,
        )
        save_cross_results(cross_result, output_dir)
        statistics = {
            "paired_test": paired_cross_operator_test(
                cross_result, int(evaluation["minimum_subjects_for_test"])
            ),
            "per_operator": per_operator_tests(
                cross_result, int(evaluation["minimum_subjects_for_test"])
            ),
        }
        write_json(output_dir / "statistics.json", statistics)
        results["cross_operator"] = cross_result
        results["statistics"] = statistics
    runtime = _runtime_metadata(start_time)
    write_json(output_dir / "runtime.json", runtime)
    results["runtime"] = runtime
    if operator_result is not None and cross_result is not None:
        write_report(
            output_dir,
            dataset,
            operator_result,
            cross_result,
            results["statistics"],
            runtime,
        )
    write_json(output_dir / "run_summary.json", results)
    return results
