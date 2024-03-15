from pathlib import Path
from typing import List, Tuple, Dict, Any

from lightning.fabric import Fabric

from chrisbase.io import hr

epsilon = 1e-7


def fabric_barrier(fabric: Fabric, title: str, c='-'):
    fabric.barrier(title)
    fabric.print(hr(c=c, title=title))


class CheckpointSaver:
    def __init__(self, fabric: Fabric, output_home: str | Path, name_format: str, saving_mode: str, num_saving: int):
        self.fabric = fabric
        self.output_home = Path(output_home)
        self.name_format = name_format
        self.num_saving = num_saving
        self.sorting_rev, self.sorting_key = saving_mode.split()
        self.sorting_rev = self.sorting_rev.lower().startswith("max")
        self.saving_checkpoints: List[Tuple[float, Path]] = []
        self.best_model_path = None

    def save_checkpoint(self, metrics: Dict[str, Any], state: Dict[str, Any]):
        ckpt_key = metrics[self.sorting_key]
        ckpt_path = self.output_home / f"{self.name_format.format(**metrics)}.ckpt"
        self.saving_checkpoints.append((ckpt_key, ckpt_path))
        self.saving_checkpoints.sort(key=lambda x: x[0], reverse=self.sorting_rev)
        for _, path in self.saving_checkpoints[self.num_saving:]:
            path.unlink(missing_ok=True)
        self.saving_checkpoints = self.saving_checkpoints[:self.num_saving]
        if (ckpt_key, ckpt_path) in self.saving_checkpoints:
            self.fabric.save(ckpt_path, state)
        self.best_model_path = self.saving_checkpoints[0][1]

    def load_checkpoint(self):
        if self.best_model_path is not None:
            self.fabric.print(f"Loading best model from {self.best_model_path}")
            return self.fabric.load(self.best_model_path)
        else:
            return None
