from dataclasses import dataclass
from typing import Any, Optional, List, Callable

import numpy as np
import torch
from seqeval import metrics as se_metrics
from seqeval import scheme as se_scheme
from sklearn import metrics as sk_metrics


@torch.no_grad()
def accuracy(preds, labels, ignore_index=None):
    assert preds.shape[0] == len(labels)
    correct = torch.sum(preds == labels)
    total = torch.sum(torch.ones_like(labels))
    if ignore_index is not None:
        # 모델이 맞춘 것 가운데 ignore index에 해당하는 것 제외
        correct -= torch.sum(torch.logical_and(preds == ignore_index, preds == labels))
        # accuracy의 분모 가운데 ignore index에 해당하는 것 제외
        total -= torch.sum(labels == ignore_index)
    return correct.to(dtype=torch.float) / total.to(dtype=torch.float)


class BasicMetricTool(torch.nn.Module):
    """Base class for metrics."""

    def __init__(
            self,
            metric_fn: Callable,
    ) -> None:
        super().__init__()
        self.preds = []
        self.targets = []
        self.metric_fn = metric_fn

    def reset(self) -> None:
        self.preds = []
        self.targets = []

    def update(self, preds, targets) -> None:
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor | float | int:
        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets)
        return score


class LabelMetricTool(BasicMetricTool):
    """Metrics requiring label information."""

    def __init__(
            self,
            metric_fn: Callable,
    ) -> None:
        super().__init__(metric_fn=metric_fn)
        self.label_info = None

    def update(self, preds, targets, label_info: Optional[Any] = None) -> None:
        super().update(preds, targets)
        if self.label_info is None:
            self.label_info = label_info

    def compute(self) -> torch.Tensor | float | int:
        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets, self.label_info)
        return score


@dataclass
class DPResult:
    heads: torch.Tensor
    types: torch.Tensor

    def __repr__(self):
        return f"""DPResult(
 - head={self.heads.tolist()},
 - types={self.types.tolist()})"""


def klue_ner_entity_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
    """KLUE-NER entity-level macro F1 (except O tag)"""
    preds = np.array(preds).flatten().tolist()
    labels = np.array(labels).flatten().tolist()
    preds_label = []
    labels_label = []

    for pred in preds:
        preds_label.append(label_list[pred])
    for label in labels:
        labels_label.append(label_list[label])

    entity_macro_f1 = se_metrics.f1_score([labels_label], [preds_label], average="macro", mode="strict", scheme=se_scheme.IOB2)
    return entity_macro_f1 * 100.0


def klue_ner_char_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
    """KLUE-NER character level macro f1 (except O tag)"""
    label_indices = list(range(len(label_list)))
    preds = np.array(preds).flatten().tolist()
    trues = np.array(labels).flatten().tolist()
    return sk_metrics.f1_score(trues, preds, labels=label_indices, average="macro", zero_division=True) * 100.0


def klue_dp_uas_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    return sk_metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="macro") * 100.0


def klue_dp_uas_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP UAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    return sk_metrics.f1_score(head_labels.tolist(), head_preds.tolist(), average="micro") * 100.0


def klue_dp_las_macro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS macro f1. (UAS : head correct / LAS : head + type correct)"""
    # UAS : head correct / LAS : head + type correct
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return sk_metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="macro") * 100.0


def klue_dp_las_micro_f1(preds: List[List[DPResult]], labels: List[List[DPResult]]) -> Any:
    """KLUE-DP LAS micro f1. (UAS : head correct / LAS : head + type correct)"""
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds[0], labels[0]):
        head_preds += pred.heads.cpu().flatten().tolist()
        head_labels += label.heads.cpu().flatten().tolist()
        type_preds += pred.types.cpu().flatten().tolist()
        type_labels += label.types.cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, (pred, label) in enumerate(zip(type_preds, type_labels)):
        if pred >= others_idx:
            type_preds[i] = -3
        if label >= others_idx:
            type_labels[i] = -3

    # pad wrong UAS
    PAD = -2
    uas_correct = np.equal(head_preds, head_labels)
    uas_incorrect = np.nonzero(np.invert(uas_correct))
    for idx in uas_incorrect:
        type_preds[idx] = PAD
    return sk_metrics.f1_score(type_labels.tolist(), type_preds.tolist(), average="micro") * 100.0


DP_UAS_MacroF1 = BasicMetricTool(klue_dp_uas_macro_f1)
DP_UAS_MicroF1 = BasicMetricTool(klue_dp_uas_micro_f1)
DP_LAS_MacroF1 = BasicMetricTool(klue_dp_las_macro_f1)
DP_LAS_MicroF1 = BasicMetricTool(klue_dp_las_micro_f1)
NER_Char_MacroF1 = LabelMetricTool(klue_ner_char_macro_f1)
NER_Entity_MacroF1 = LabelMetricTool(klue_ner_entity_macro_f1)
