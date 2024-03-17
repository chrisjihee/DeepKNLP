import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import MLArguments

logger = logging.getLogger(__name__)


@dataclass
class ClassificationExample:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ClassificationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[int] = None


class NsmcCorpus:

    def __init__(self, args: MLArguments):
        self.args = args

    @property
    def num_labels(self):
        return len(self.get_labels())

    def get_labels(self):
        return ["0", "1"]

    def read_raw_examples(self, split: str) -> List[ClassificationExample]:
        assert self.args.data.home, f"No data_home: {self.args.data.home}"
        assert self.args.data.name, f"No data_name: {self.args.data.name}"
        data_file_dict: dict = self.args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {self.args.data.files}"
        data_path: Path = Path(self.args.data.home) / self.args.data.name / data_file_dict[split]
        assert data_path.exists() and data_path.is_file(), f"No data_text_path: {data_path}"
        if self.args.prog.local_rank == 0:
            logger.info(f"Creating features from {data_path}")
        lines = list(csv.reader(open(data_path, "r", encoding="utf-8"), delimiter="\t", quotechar='"'))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            _, text_a, label = line
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        if self.args.prog.local_rank == 0:
            logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    def raw_examples_to_encoded_examples(
            self,
            raw_examples: List[ClassificationExample],
            tokenizer: PreTrainedTokenizer,
            label_list: List[str],
    ) -> List[ClassificationFeatures]:
        label_to_id: Dict[str, int] = {label: i for i, label in enumerate(label_list)}
        labels: List[int] = [label_to_id[example.label] for example in raw_examples]

        if raw_examples[0].text_b is not None:
            batch_text_or_text_pairs = [(example.text_a, example.text_b) for example in raw_examples if example.text_b is not None]
        else:
            batch_text_or_text_pairs = [example.text_a for example in raw_examples if example.text_a is not None]
        batch_encoding: BatchEncoding = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs,
            max_length=self.args.model.seq_len,
            padding="max_length",
            truncation=True,
        )

        encoded_examples = []
        for i in range(len(raw_examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = ClassificationFeatures(**inputs, label=labels[i])
            encoded_examples.append(feature)

        if self.args.prog.local_rank == 0:
            for i, raw_example in enumerate(raw_examples[: self.args.data.num_check]):
                logger.info("  === [Example %d] ===" % (i + 1))
                if raw_example.text_b is None:
                    logger.info("  = sentence : %s" % (raw_example.text_a))
                else:
                    logger.info("  = sentence : %s" % (raw_example.text_a + " + " + raw_example.text_b))
                logger.info("  = tokens   : %s" % " ".join(batch_encoding.tokens(i)))
                logger.info("  = label    : %s" % (raw_example.label))
                logger.info("  = features : %s" % encoded_examples[i])
                logger.info("  === ")
            logger.info(f"Converted {len(raw_examples)} raw examples to {len(encoded_examples)} encoded examples")
        return encoded_examples


class ClassificationDataset(Dataset):

    def __init__(self, split: str, tokenizer: PreTrainedTokenizer, data: NsmcCorpus):
        self.data: NsmcCorpus = data
        examples: List[ClassificationExample] = self.data.read_raw_examples(split)
        self.label_list: List[str] = self.data.get_labels()
        self.features: List[ClassificationFeatures] = self.data.raw_examples_to_encoded_examples(
            examples, tokenizer, label_list=self.label_list)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self) -> List[str]:
        return self.label_list
