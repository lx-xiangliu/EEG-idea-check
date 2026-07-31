from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.data.synthetic import SyntheticBundle
from src.losses import AlignmentObjective, linear_cka
from src.models import AudioTeacher, DepthDerivativeExtractor, EEGEncoder
from src.utils.repro import assert_finite, seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class RunResult:
    seed: int
    mode: str
    method: str
    probe_accuracy: float
    probe_r2: float
    nuisance_accuracy: float
    final_cka: float
    residual_cka: float
    mapping_rmse: float | None
    best_val_loss: float
    trainable_parameters: int
    epochs: int
    wall_time_seconds: float
    mapping_means: list[float]
    mapping_weights: list[list[float]]
    eeg_residual_norms: list[float]
    audio_residual_norms: list[float]
    hidden_cka_matrix: list[list[float]]
    residual_cka_matrix: list[list[float]]
    layer_probe_accuracy: list[float]
    residual_probe_accuracy: list[float]


class SyntheticAlignmentModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eeg = EEGEncoder(
            input_dim=cfg.data.eeg_dim,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.eeg_layers,
            n_heads=cfg.model.n_heads,
            ff_mult=cfg.model.ff_mult,
            dropout=cfg.model.dropout,
            max_len=cfg.data.seq_len,
        )
        teacher_hierarchy = cfg.data.mode
        if cfg.train.method == "random_teacher":
            teacher_hierarchy = "parallel"
        self.audio = AudioTeacher(
            input_dim=cfg.data.audio_dim,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.audio_layers,
            hierarchy=teacher_hierarchy,
            seed=10_000 + cfg.train.seed,
        )
        self.objective = AlignmentObjective(
            method=cfg.train.method,
            d_model=cfg.model.d_model,
            projection_dim=cfg.model.projection_dim,
            eeg_layers=cfg.model.eeg_layers,
            audio_layers=cfg.model.audio_layers,
            normalize_residuals=cfg.train.normalize_residuals,
            mapper_temperature=cfg.model.mapper_temperature,
        )

    def forward(self, eeg: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, Any]:
        eeg_output = self.eeg(eeg, return_hidden_states=True)
        with torch.no_grad():
            audio_output = self.audio(audio, return_hidden_states=True)
        audio_states = audio_output["hidden_states"]
        if self.cfg.train.method == "random_teacher":
            permutation = torch.arange(audio.shape[0] - 1, -1, -1, device=audio.device)
            audio_states = [state[permutation] for state in audio_states]
        return self.objective(eeg_output["hidden_states"], audio_states, eeg_input=eeg)


def _loader(dataset: TensorDataset, cfg: ExperimentConfig, shuffle: bool, offset: int) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.train.seed + offset)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def _mean_loss(model: SyntheticAlignmentModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for eeg, audio, _, _ in loader:
        eeg, audio = eeg.to(device), audio.to(device)
        loss, _ = model(eeg, audio)
        total += float(loss.item()) * eeg.shape[0]
        count += eeg.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def _extract(
    model: SyntheticAlignmentModel, dataset: TensorDataset, device: torch.device
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    eeg_states_all: list[list[torch.Tensor]] = []
    audio_states_all: list[list[torch.Tensor]] = []
    latents: list[torch.Tensor] = []
    subjects: list[torch.Tensor] = []
    diagnostics = None
    model.eval()
    for eeg, audio, z, subject in loader:
        eeg, audio = eeg.to(device), audio.to(device)
        eeg_states = model.eeg(eeg, return_hidden_states=True)["hidden_states"]
        audio_states = model.audio(audio, return_hidden_states=True)["hidden_states"]
        _, diagnostics = model.objective(eeg_states, audio_states, eeg_input=eeg)
        eeg_states_all.append([state.cpu() for state in eeg_states])
        audio_states_all.append([state.cpu() for state in audio_states])
        latents.append(z)
        subjects.append(subject)
    eeg_states = [torch.cat([batch[layer] for batch in eeg_states_all], dim=0) for layer in range(len(eeg_states_all[0]))]
    audio_states = [torch.cat([batch[layer] for batch in audio_states_all], dim=0) for layer in range(len(audio_states_all[0]))]
    return {
        "eeg_states": eeg_states,
        "audio_states": audio_states,
        "z": torch.cat(latents).numpy(),
        "subject": torch.cat(subjects).numpy(),
        "diagnostics": diagnostics,
    }


def _probe_accuracy(train_x: np.ndarray, train_z: np.ndarray, test_x: np.ndarray, test_z: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(train_z.shape[1]):
        train_y = (train_z[:, index] > 0).astype(np.int64)
        test_y = (test_z[:, index] > 0).astype(np.int64)
        classifier = LogisticRegression(max_iter=500, random_state=0)
        classifier.fit(train_x, train_y)
        scores.append(float(accuracy_score(test_y, classifier.predict(test_x))))
    return float(np.mean(scores))


def _probe_r2(train_x: np.ndarray, train_z: np.ndarray, test_x: np.ndarray, test_z: np.ndarray) -> float:
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_z)
    return float(r2_score(test_z, model.predict(test_x), multioutput="uniform_average"))


def _nuisance_accuracy(x: np.ndarray, subject: np.ndarray, seed: int) -> float:
    counts = np.bincount(subject)
    valid = np.flatnonzero(counts >= 2)
    keep = np.isin(subject, valid)
    x, subject = x[keep], subject[keep]
    if len(np.unique(subject)) < 2:
        return float("nan")
    left, right = train_test_split(
        np.arange(len(subject)), test_size=0.5, random_state=seed, stratify=subject
    )
    classifier = LogisticRegression(max_iter=500, random_state=seed)
    classifier.fit(x[left], subject[left])
    return float(accuracy_score(subject[right], classifier.predict(x[right])))


def _matrix_cka(left_states: list[torch.Tensor], right_states: list[torch.Tensor]) -> np.ndarray:
    matrix = np.zeros((len(left_states), len(right_states)), dtype=np.float64)
    for i, left in enumerate(left_states):
        for j, right in enumerate(right_states):
            matrix[i, j] = float(linear_cka(left, right).item())
    return matrix


def _layer_probes(train: dict[str, Any], test: dict[str, Any]) -> tuple[list[float], list[float]]:
    hidden_scores = []
    for train_state, test_state in zip(train["eeg_states"], test["eeg_states"]):
        hidden_scores.append(
            _probe_accuracy(
                train_state.mean(dim=1).numpy(),
                train["z"],
                test_state.mean(dim=1).numpy(),
                test["z"],
            )
        )
    extractor = DepthDerivativeExtractor()
    train_residuals = extractor(train["eeg_states"], normalize=True)
    test_residuals = extractor(test["eeg_states"], normalize=True)
    residual_scores = []
    for train_state, test_state in zip(train_residuals, test_residuals):
        residual_scores.append(
            _probe_accuracy(
                train_state.mean(dim=1).numpy(),
                train["z"],
                test_state.mean(dim=1).numpy(),
                test["z"],
            )
        )
    return hidden_scores, residual_scores


def _evaluate(
    model: SyntheticAlignmentModel,
    bundle: SyntheticBundle,
    cfg: ExperimentConfig,
    best_val_loss: float,
    wall_time: float,
) -> RunResult:
    device = torch.device(cfg.train.device)
    train = _extract(model, bundle.train, device)
    test = _extract(model, bundle.test, device)
    train_final = train["eeg_states"][-1].mean(dim=1).numpy()
    test_final = test["eeg_states"][-1].mean(dim=1).numpy()
    probe_accuracy = _probe_accuracy(train_final, train["z"], test_final, test["z"])
    probe_r2 = _probe_r2(train_final, train["z"], test_final, test["z"])
    nuisance = _nuisance_accuracy(test_final, test["subject"], cfg.train.seed)
    hidden_cka = _matrix_cka(test["eeg_states"], test["audio_states"])
    extractor = DepthDerivativeExtractor()
    eeg_residuals = extractor(test["eeg_states"], normalize=True)
    audio_residuals = extractor(test["audio_states"], normalize=True)
    residual_cka = _matrix_cka(eeg_residuals, audio_residuals)
    diagonal_hidden = np.mean(
        [hidden_cka[i, round(i * (hidden_cka.shape[1] - 1) / max(hidden_cka.shape[0] - 1, 1))] for i in range(hidden_cka.shape[0])]
    )
    diagonal_residual = np.mean(
        [residual_cka[i, round(i * (residual_cka.shape[1] - 1) / max(residual_cka.shape[0] - 1, 1))] for i in range(residual_cka.shape[0])]
    )
    diagnostics = test["diagnostics"]
    means: list[float] = []
    weights: list[list[float]] = []
    mapping_rmse: float | None = None
    if diagnostics is not None and diagnostics.means is not None:
        means = diagnostics.means.detach().cpu().tolist()
        target = np.linspace(0, cfg.model.audio_layers - 1, cfg.model.eeg_layers)
        mapping_rmse = float(np.sqrt(np.mean((np.asarray(means) - target) ** 2)))
    if diagnostics is not None and diagnostics.weights is not None:
        weights = diagnostics.weights.detach().cpu().tolist()
    layer_probe, residual_probe = _layer_probes(train, test)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return RunResult(
        seed=cfg.train.seed,
        mode=cfg.data.mode,
        method=cfg.train.method,
        probe_accuracy=probe_accuracy,
        probe_r2=probe_r2,
        nuisance_accuracy=nuisance,
        final_cka=float(diagonal_hidden),
        residual_cka=float(diagonal_residual),
        mapping_rmse=mapping_rmse,
        best_val_loss=best_val_loss,
        trainable_parameters=trainable_parameters,
        epochs=cfg.train.epochs,
        wall_time_seconds=wall_time,
        mapping_means=means,
        mapping_weights=weights,
        eeg_residual_norms=[float(value.norm(dim=-1).mean().item()) for value in eeg_residuals],
        audio_residual_norms=[float(value.norm(dim=-1).mean().item()) for value in audio_residuals],
        hidden_cka_matrix=hidden_cka.tolist(),
        residual_cka_matrix=residual_cka.tolist(),
        layer_probe_accuracy=layer_probe,
        residual_probe_accuracy=residual_probe,
    )


def train_synthetic(
    cfg: ExperimentConfig,
    bundle: SyntheticBundle,
    save_checkpoint: bool = True,
) -> RunResult:
    seed_everything(cfg.train.seed)
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing silent CPU fallback")
    model = SyntheticAlignmentModel(cfg).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    train_loader = _loader(bundle.train, cfg, shuffle=True, offset=101)
    val_loader = _loader(bundle.val, cfg, shuffle=False, offset=202)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()
    for epoch in range(cfg.train.epochs):
        model.train()
        for eeg, audio, _, _ in train_loader:
            eeg, audio = eeg.to(device), audio.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(eeg, audio)
            assert_finite("training loss", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.train.grad_clip)
            optimizer.step()
        val_loss = _mean_loss(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        LOGGER.debug("epoch=%d val_loss=%.6f", epoch, val_loss)
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    wall_time = time.perf_counter() - start
    if save_checkpoint:
        checkpoint_dir = Path(cfg.train.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{cfg.data.mode}_{cfg.train.method}_seed{cfg.train.seed}.pt"
        torch.save(
            {"model": best_state, "config": cfg.to_dict(), "best_val_loss": best_loss},
            checkpoint_path,
        )
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {"config": cfg.to_dict(), "best_val_loss": best_loss},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return _evaluate(model, bundle, cfg, best_loss, wall_time)
