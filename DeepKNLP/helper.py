from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import List, Dict, Any

import torch
from lightning.fabric import Fabric

from chrisbase.io import hr

epsilon = 1e-7


@dataclass
class ScorePath:
    score: float
    path: Path


class CheckpointSaver:
    def __init__(self, fabric: Fabric, output_home: str | Path, name_format: str, saving_mode: str, num_saving: int):
        self.fabric = fabric
        self.output_home = Path(output_home)
        self.name_format = name_format
        self.num_saving = num_saving
        self.sort_reverse, self.score_key = saving_mode.split()
        self.sort_reverse = self.sort_reverse.lower().startswith("max")
        self.best_model_path: str | Path | None = None
        self.saving_checkpoints: List[ScorePath] = []

    def save_checkpoint(self, metrics: Dict[str, Any], ckpt_state: Dict[str, Any]):
        ckpt_item = ScorePath(
            score=metrics[self.score_key],
            path=self.output_home / f"{self.name_format.format(**metrics)}.ckpt",
        )
        self.saving_checkpoints.append(ckpt_item)
        self.saving_checkpoints.sort(key=attrgetter('score'),
                                     reverse=self.sort_reverse)
        for i in range(self.num_saving, len(self.saving_checkpoints)):
            self.saving_checkpoints.pop(i).path.unlink(missing_ok=True)
        if ckpt_item in self.saving_checkpoints:
            self.fabric.save(ckpt_item.path, ckpt_state)
        self.best_model_path = self.saving_checkpoints[0].path


def fabric_barrier(fabric: Fabric, title: str, c='-'):
    fabric.barrier(title)
    fabric.print(hr(c=c, title=title))


def data_collator(features):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch
