#!/usr/bin/env python3
"""Build a reproducible novelty matrix from cached OpenAlex records plus audited primary pages."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from _common import ROOT


CURATED: dict[str, list[str]] = {
    "trf_speech_eeg": [
        "Modulation Spectra Capture EEG Responses to Speech Signals and Drive Distinct Temporal Response Functions",
        "Predictors for estimating subcortical EEG responses to continuous speech",
        "Neural Markers of Speech Comprehension: Measuring EEG Tracking of Linguistic Speech Representations, Controlling the Speech Acoustics",
        "Eelbrain, a Python toolkit for time-continuous analysis with temporal response functions",
        "Analyzing EEG Signals in Auditory Speech Comprehension Using Temporal Response Functions and Generalized Additive Models",
        "Neural Speech Tracking in the Theta and in the Delta Frequency Band Differentially Encode Clarity and Comprehension of Speech in Noise",
        "Prediction of Speech Intelligibility by Means of EEG Responses to Sentences in Noise",
        "EEG-based auditory attention detection: boundary conditions for background noise and speaker positions*",
    ],
    "eeg_audio": [
        "Decoding Covert Speech From EEG-A Comprehensive Review",
        "Decoding speech perception from non-invasive brain recordings",
        "Comparison of Two-Talker Attention Decoding from EEG with Nonlinear Neural Networks and Linear Methods",
        "STAnet: A Spatiotemporal Attention Network for Decoding Auditory Spatial Attention From EEG",
        "Relating EEG to continuous speech using deep neural networks: a review",
        "Robust decoding of the speech envelope from EEG recordings through deep neural networks",
        "EEG-based detection of the locus of auditory attention with convolutional neural networks",
        "Electroencephalography-Based Auditory Attention Decoding: Toward Neurosteered Hearing Devices",
        "Neural decoding of music from the EEG",
        "A neural speech decoding framework leveraging deep learning and speech synthesis",
    ],
    "nuisance_projection": [
        "Orthogonal statistical learning",
        "Unsupervised Learning with Contrastive Latent Variable Models",
        "Orthogonal Projection Loss",
        "Disentangled Representation Learning",
        "Learning Robust Representations via Multi-View Information Bottleneck",
    ],
    "attention_deconfounding": [
        "Deconfounded Video Moment Retrieval with Causal Intervention",
        "Deconfounded Visual Grounding",
        "Towards Deconfounded Image-Text Matching with Causal Inference",
        "Knowledge Proxy Intervention for Deconfounded Video Question Answering",
        "Causal Attention for Vision-Language Tasks",
        "Video-Audio Domain Generalization via Confounder Disentanglement",
        "Backdoor Defense via Deconfounded Representation Learning",
    ],
    "brain_variance_partition": [
        "Semantic Context Enhances the Early Auditory Encoding of Natural Speech",
        "Shared computational principles for language processing in humans and deep language models",
        "Low-frequency cortical responses to natural speech reflect probabilistic phonotactics",
        "Exploring neural tracking of acoustic and linguistic speech representations in individuals with post‐stroke aphasia",
        "Linguistic Structure and Meaning Organize Neural Oscillations into a Content-Specific Hierarchy",
        "Modulation of brain activity by psycholinguistic information during naturalistic speech comprehension and production",
        "Neural tracking measures of speech intelligibility: Manipulating intelligibility while keeping acoustics unchanged",
        "Shared functional specialization in transformer-based language models and the human brain",
    ],
}


MANUAL = [
    ("nuisance_projection", "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection", "Shauli Ravfogel et al.", 2020, "ACL", "https://aclanthology.org/2020.acl-main.647/", "Iterative null-space projection removes linearly decodable attributes; close statistical ancestor, not EEG or attention."),
    ("nuisance_projection", "Better Hit the Nail on the Head than Beat around the Bush: Removing Protected Attributes with a Single Projection", "Pantea Haghighatkhah et al.", 2022, "EMNLP", "https://aclanthology.org/2022.emnlp-main.575/", "Single targeted projection and random-projection control directly motivate TPA specificity tests."),
    ("nuisance_projection", "LEACE: Perfect linear concept erasure in closed form", "Nora Belrose et al.", 2023, "NeurIPS", "https://openreview.net/forum?id=awIpKpwTwF", "Closed-form least-squares concept erasure shows projection is established and may be oblique rather than orthogonal."),
    ("nuisance_projection", "Invariant Representations without Adversarial Training", "Daniel Moyer et al.", 2018, "NeurIPS", "https://proceedings.neurips.cc/paper/2018/hash/415185ea244ea2b2bedeb0449b926802-Abstract.html", "Information-theoretic nuisance invariance; stronger than claiming linear residualization estimates conditional MI."),
    ("nuisance_projection", "Learning Invariant Representations with Missing Data", "Mark Goldstein et al.", 2022, "CLeaR", "https://proceedings.mlr.press/v177/goldstein22a.html", "Observed nuisance labels and missingness matter for invariant representation guarantees."),
    ("nuisance_projection", "Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation", "Floris Holstege; Bram Wouters; Noud Van Giersbergen; Cees Diks", 2024, "ICML", "https://proceedings.mlr.press/v235/holstege24a.html", "Jointly estimates orthogonal task and spurious subspaces; close control for random-subspace explanations."),
    ("nuisance_projection", "Statistical Learning with a Nuisance Component", "Dylan Foster and Vasilis Syrgkanis", 2019, "COLT", "https://proceedings.mlr.press/v99/foster19c.html", "Formal nuisance-estimation error propagation; relevant to batch-estimated C but not an attention operator."),
    ("nuisance_projection", "Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE", "Alicja Dobrzeniecka et al.", 2025, "Findings ACL", "https://aclanthology.org/2025.findings-acl.674/", "Shows INLP can introduce random modifications and evaluates more targeted erasure."),
    ("conditional_multimodal", "Conditional Contrastive Learning with Kernel", "Yao-Hung Hubert Tsai et al.", 2022, "ICLR", "https://openreview.net/forum?id=AAJLBoGt0XM", "Direct conditional contrastive prior work; conditions sampling/objective rather than residualizing Q/K."),
    ("conditional_multimodal", "Conditional Contrastive Networks", "Emily Mu and John Guttag", 2022, "NeurIPS workshop", "https://openreview.net/forum?id=MbpMqAXAGFH", "Learns separate conditional similarity subspaces; no lagged acoustic nuisance or EEG."),
    ("conditional_multimodal", "Conditional Mutual Information for Disentangled Representations in Reinforcement Learning", "Mhairi Dunion et al.", 2023, "NeurIPS", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/fd750154df5f199f94df897975621306-Abstract-Conference.html", "Actually optimizes a CMI-motivated objective, underscoring that an attention score is not a CMI estimator."),
    ("conditional_multimodal", "ContraGAN: Contrastive Learning for Conditional Image Generation", "Minguk Kang and Jaesik Park", 2020, "NeurIPS", "https://proceedings.neurips.cc/paper/2020/hash/f490c742cd8318b8ee6dca10af2a163f-Abstract.html", "Class-conditional contrastive loss; semantic overlap in name only."),
    ("conditional_multimodal", "Multimodal Contrastive Learning via Uni-Modal Coding and Cross-Modal Prediction for Multimodal Sentiment Analysis", "Ronghao Lin and Haifeng Hu", 2022, "Findings EMNLP", "https://aclanthology.org/2022.findings-emnlp.36/", "Cross-modal contrastive baseline without nuisance conditioning."),
    ("conditional_multimodal", "InfoGCL: Information-Aware Graph Contrastive Learning", "Dongkuan Xu et al.", 2021, "NeurIPS", "https://proceedings.neurips.cc/paper/2021/hash/ff1e68e74c6b16a1a7b5d958b95e120c-Abstract.html", "Information bottleneck comparison; no EEG or Q/K residualization."),
    ("conditional_multimodal", "Information Subtraction: Learning Representations for Conditional Entropy", "Keng-Hou Leong et al.", 2024, "ICLR submission", "https://openreview.net/forum?id=C2uViDZmNp", "Targets arbitrary continuous conditioning via learned information subtraction, broader but less established."),
    ("conditional_multimodal", "Enhancing Multimodal Entity Linking with Jaccard Distance-based Conditional Contrastive Learning and Contextual Visual Augmentation", "Cong-Duy T Nguyen et al.", 2025, "NAACL", "https://aclanthology.org/2025.naacl-long.341/", "Condition-matched hard negatives are conceptually close to acoustically matched negatives."),
    ("conditional_multimodal", "Conditional Semantic Textual Similarity via Conditional Contrastive Learning", "Xinyue Liu et al.", 2025, "COLING", "https://aclanthology.org/2025.coling-main.306/", "Conditional similarity objective, not covariate residualization."),
    ("conditional_multimodal", "Conditional Noise-Contrastive Estimation of Unnormalised Models", "Ciwan Ceylan and Michael Gutmann", 2018, "ICML", "https://proceedings.mlr.press/v80/ceylan18a.html", "Conditional NCE is density estimation and not equivalent to partial attention."),
    ("brain_variance_partition", "Speech language models lack important brain-relevant semantics", "Subba Reddy Oota et al.", 2024, "ACL", "https://aclanthology.org/2024.acl-long.462/", "Closest mechanism paper: explicitly removes low-level features before brain alignment and reports loss of late-region predictive power for speech models."),
    ("trf_speech_eeg", "Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words", "Jin Dou; Andrew J. Anderson; Aaron S. White; Samuel V. Norman-Haignere; Edmund C. Lalor", 2025, "PLOS Computational Biology", "https://doi.org/10.1371/journal.pcbi.1013006", "Dynamic amplitude/latency TRF with envelope/onset controls; shows fixed lag is a limitation."),
    ("trf_speech_eeg", "The effects of data quantity on performance of temporal response function analyses of natural speech processing", "Juraj Mesik; Magdalena Wojtczak", 2023, "Frontiers in Neuroscience", "https://www.frontiersin.org/articles/10.3389/fnins.2022.963629", "Quantifies data requirements and unique envelope/onset/surprisal contributions for TRF inference."),
    ("brain_variance_partition", "Early language experience modulates the tradeoff between acoustic-temporal and lexico-semantic cortical tracking of speech", "Jose Perez-Navarro; Anastasia Klimovich-Gray; Mikel Lizarazu; Giorgio Piazza; Nicola Molinaro; Marie Lallier", 2024, "iScience", "https://doi.org/10.1016/j.isci.2024.110247", "Joint envelope and semantic TRFs support measurable higher-level EEG effects in a specific setting."),
    ("brain_variance_partition", "Attention Modulation to Linguistic Speech Units", "Manuela Jaeger; Elana Zion-Golumbic; Martin Georg Bleichner", 2025, "Neurobiology of Language", "https://doi.org/10.1162/nol.a.14", "Phoneme/word onset TRFs remain after including the speech envelope as a regressor."),
    ("nuisance_projection", "A framework for analyzing concept representations in neural models", "Burin Naowarat et al.", 2026, "CoNLL", "https://aclanthology.org/2026.conll-main.34/", "HuBERT phone/speaker subspaces show erasure generalization and containment failures."),
]


def inverted_abstract(record: dict[str, object]) -> str:
    index = record.get("abstract_inverted_index") or {}
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((int(position), word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def api_records() -> list[dict[str, object]]:
    title_to_category = {title: category for category, titles in CURATED.items() for title in titles}
    found: dict[str, dict[str, object]] = {}
    for path in (ROOT / "reports" / "source_metadata" / "openalex").glob("*_lit.json"):
        for record in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if record["title"] in title_to_category:
                found[record["title"]] = record
    missing = set(title_to_category) - set(found)
    if missing:
        raise RuntimeError(f"curated OpenAlex records missing: {sorted(missing)}")
    rows = []
    for title, record in found.items():
        location = record.get("primary_location") or {}
        source = location.get("source") or {}
        authors = "; ".join(a["author"]["display_name"] for a in record.get("authorships", []))
        rows.append(base_row(
            category=title_to_category[title], title=title, authors=authors,
            year=record["publication_year"], venue=source.get("display_name", ""),
            url=location.get("landing_page_url") or record.get("doi") or record["id"],
            notes=inverted_abstract(record)[:420].replace("\n", " "),
            paper_id=record["id"].rsplit("/", 1)[-1],
        ))
    return rows


def base_row(category: str, title: str, authors: str, year: int, venue: str, url: str, notes: str, paper_id: str | None = None) -> dict[str, object]:
    lower = (title + " " + notes).lower()
    is_eeg = "eeg" in lower or category in {"trf_speech_eeg", "eeg_audio"}
    is_trf = "temporal response" in lower or "trf" in lower
    projection = category == "nuisance_projection" or any(token in lower for token in ["projection", "residual", "erasure", "invariant"])
    condition = category in {"conditional_multimodal", "attention_deconfounding"}
    acoustic = any(token in lower for token in ["acoustic", "speech envelope", "speech acoustics"])
    qk_direct = all(token in lower for token in ["query", "key", "residual"])
    return {
        "paper_id": paper_id or f"manual-{hashlib.sha1(f'{title}|{year}'.encode()).hexdigest()[:10]}",
        "title": title, "authors": authors, "year": year, "venue": venue,
        "task": category, "modalities": "EEG-audio" if is_eeg else ("multimodal" if condition else "representation"),
        "trf_or_lagged_design": is_trf, "partial_correlation": "partial" in lower,
        "residualize_input": projection, "residualize_embedding": projection,
        "residualize_q": qk_direct, "residualize_k": qk_direct, "residualize_v": False,
        "condition_attention": category == "attention_deconfounding", "acoustic_covariates": acoustic,
        "brain_data": is_eeg or category == "brain_variance_partition", "teacher_model": "language model" in lower or "hubert" in lower,
        "hard_negative_control": "hard negative" in lower or "matched negative" in lower,
        "code_available": "code" in lower, "closest_overlap": category,
        "novelty_risk": "high" if category in {"nuisance_projection", "brain_variance_partition"} else "medium",
        "url": url, "notes": notes,
    }


def main() -> None:
    rows = api_records()
    rows.extend(base_row(*item) for item in MANUAL)
    frame = pd.DataFrame(rows).drop_duplicates(subset=["title"]).sort_values(["task", "year", "title"])
    output = ROOT / "reports" / "novelty_matrix.csv"
    frame.to_csv(output, index=False)
    counts = frame.groupby("task").size().sort_index().to_dict()
    print(f"wrote {len(frame)} unique papers to {output}")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
