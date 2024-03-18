import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, ClassVar

import torch
from dataclasses_json import DataClassJsonMixin
from torch.utils.data.dataset import Dataset
from transformers import CharSpan
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from DeepKNLP.arguments import MLArguments
from chrisbase.io import hr, LoggingFormat, setup_unit_logger
from chrisbase.io import make_parent_dir, merge_dicts

logger = logging.getLogger(__name__)
setup_unit_logger(fmt=LoggingFormat.CHECK_24)


@dataclass
class EntityInText(DataClassJsonMixin):
    pattern: ClassVar[re.Pattern] = re.compile('<(?P<text>[^<>]+?):(?P<label>[A-Z]{2,3})>')
    text: str
    label: str
    offset: tuple[int, int]

    @staticmethod
    def from_match(m: re.Match, s: str) -> tuple["EntityInText", str]:
        g = m.groupdict()
        x = g['text']
        y = g['label']
        z = (m.start(), m.start() + len(x))
        e = EntityInText(text=x, label=y, offset=z)
        s = s[:m.start()] + x + s[m.end():]
        return e, s

    def to_offset_lable_dict(self) -> Dict[int, str]:
        offset_list = [(self.offset[0], f"B-{self.label}")]
        for i in range(self.offset[0] + 1, self.offset[1]):
            offset_list.append((i, f"I-{self.label}"))
        return dict(offset_list)


@dataclass
class NERTaggedExample(DataClassJsonMixin):
    example_id: str = field(default_factory=str)
    origin: str = field(default_factory=str)
    tagged: str = field(default_factory=str)

    @classmethod
    def from_tsv(cls, tsv: str):
        lines = tsv.strip().splitlines()
        meta = [x.split('\t') for x in lines if x.startswith('#')][-1]
        chars = [x.split('\t') for x in lines if not x.startswith('#')]
        example_id = re.sub(r"^##+", "", meta[0]).strip()
        tagged = meta[1].strip()
        origin = ''.join(x[0] for x in chars)
        return cls(example_id=example_id, origin=origin, tagged=tagged)


@dataclass
class NERParsedExample(DataClassJsonMixin):
    origin: str = field(default_factory=str)
    entity_list: List[EntityInText] = field(default_factory=list)
    character_list: List[tuple[str, str]] = field(default_factory=list)

    def get_offset_label_dict(self):
        return {i: y for i, (_, y) in enumerate(self.character_list)}

    def to_tagged_text(self, entity_form=lambda e: f"<{e.text}:{e.label}>"):
        self.entity_list.sort(key=lambda x: x.offset[0])
        cursor = 0
        tagged_text = ""
        for e in self.entity_list:
            tagged_text += self.origin[cursor: e.offset[0]] + entity_form(e)
            cursor = e.offset[1]
        tagged_text += self.origin[cursor:]
        return tagged_text

    @classmethod
    def from_tagged(cls, origin: str, tagged: str, debug: bool = False) -> Optional["NERParsedExample"]:
        entity_list: List[EntityInText] = []
        if debug:
            logging.debug(f"* origin: {origin}")
            logging.debug(f"  tagged: {tagged}")
        restored = tagged[:]
        no_problem = True
        offset_labels = {i: "O" for i in range(len(origin))}
        while True:
            match: re.Match = EntityInText.pattern.search(restored)
            if not match:
                break
            entity, restored = EntityInText.from_match(match, restored)
            extracted = origin[entity.offset[0]:entity.offset[1]]
            if entity.text == extracted:
                entity_list.append(entity)
                offset_labels = merge_dicts(offset_labels, entity.to_offset_lable_dict())
            else:
                no_problem = False
            if debug:
                logging.debug(f"  = {entity} -> {extracted}")
                logging.debug(f"    {offset_labels}")
        if debug:
            logging.debug(f"  --------------------")
        character_list = [(origin[i], offset_labels[i]) for i in range(len(origin))]
        if restored != origin:
            no_problem = False
        return cls(origin=origin,
                   entity_list=entity_list,
                   character_list=character_list) if no_problem else None


@dataclass
class NEREncodedExample:
    idx: int
    raw: NERParsedExample
    encoded: BatchEncoding
    label_ids: Optional[List[int]] = None


class NERCorpus:
    def __init__(self, args: MLArguments):
        self.args = args
        self.labels: List[str] = self.extract_labels()

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @classmethod
    def extract_labels_from_data(cls,
                                 data_path: str | Path,
                                 label_path: str | Path,
                                 args: MLArguments = None) -> List[str]:
        label_path = make_parent_dir(label_path).absolute()
        data_path = Path(data_path).absolute()
        assert data_path.exists() and data_path.is_file() or label_path.exists() and label_path.is_file(), f"No data_path or label_path: {data_path}, {label_path}"
        if label_path.exists():
            labels = label_path.read_text().splitlines()
            if args and args.prog.local_rank == 0:
                logger.info(f"Loaded {len(labels)} labels from {label_path}")
        else:
            if args and args.prog.local_rank == 0:
                logger.info(f"Extracting labels from {data_path}")
            ner_tags = []
            with data_path.open() as inp:
                for line in inp.readlines():
                    for x in NERParsedExample.from_json(line).entity_list:
                        if x.label not in ner_tags:
                            ner_tags.append(x.label)
            ner_tags = sorted(ner_tags)
            b_tags = [f"B-{ner_tag}" for ner_tag in ner_tags]
            i_tags = [f"I-{ner_tag}" for ner_tag in ner_tags]
            labels = ["O"] + b_tags + i_tags
            if args and args.prog.local_rank == 0:
                logger.info(f"Saved {len(labels)} labels to {label_path}")
            with label_path.open("w") as f:
                f.writelines([x + "\n" for x in labels])
        return labels

    def extract_labels(self) -> List[str]:
        if not self.args.data or not self.args.data.files:
            if self.args and self.args.prog.local_rank == 0:
                logger.warning(f"Empty label_list (no data or data_files)")
            return []
        label_path = make_parent_dir(self.args.env.output_home.parent / f"label_map={self.args.data.name}.txt")
        train_data_path = self.args.data.home / self.args.data.name / self.args.data.files.train if self.args.data.files.train else None
        valid_data_path = self.args.data.home / self.args.data.name / self.args.data.files.valid if self.args.data.files.valid else None
        test_data_path = self.args.data.home / self.args.data.name / self.args.data.files.test if self.args.data.files.test else None
        train_data_path = train_data_path if train_data_path and train_data_path.exists() else None
        valid_data_path = valid_data_path if valid_data_path and valid_data_path.exists() else None
        test_data_path = test_data_path if test_data_path and test_data_path.exists() else None
        data_path = test_data_path or valid_data_path or train_data_path
        return self.extract_labels_from_data(data_path=data_path, label_path=label_path, args=self.args)

    def read_raw_examples(self, split: str) -> List[NERParsedExample]:
        assert self.args.data.home, f"No data_home: {self.args.data.home}"
        assert self.args.data.name, f"No data_name: {self.args.data.name}"
        data_file_dict: dict = self.args.data.files.to_dict()
        assert split in data_file_dict, f"No '{split}' split in data_file: should be one of {list(data_file_dict.keys())}"
        assert data_file_dict[split], f"No data_file for '{split}' split: {self.args.data.files}"
        data_path: Path = Path(self.args.data.home) / self.args.data.name / data_file_dict[split]
        assert data_path.exists() and data_path.is_file(), f"No data_text_path: {data_path}"
        if self.args.prog.local_rank == 0:
            logger.info(f"Creating features from {data_path}")

        examples = []
        with data_path.open(encoding="utf-8") as inp:
            for line in inp.readlines():
                examples.append(NERParsedExample.from_json(line))
        if self.args.prog.local_rank == 0:
            logger.info(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    @staticmethod
    def _decide_span_label(span: CharSpan, offset_to_label: Dict[int, str]):
        for x in [offset_to_label[i] for i in range(span.start, span.end)]:
            if x.startswith("B-") or x.startswith("I-"):
                return x
        return "O"

    def raw_examples_to_encoded_examples(
            self,
            raw_examples: List[NERParsedExample],
            tokenizer: PreTrainedTokenizerFast,
            label_list: List[str],
    ) -> List[NEREncodedExample]:
        label_to_id: Dict[str, int] = {label: i for i, label in enumerate(label_list)}
        id_to_label: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
        if self.args.prog.local_rank == 0:
            logger.debug(f"label_to_id = {label_to_id}")
            logger.debug(f"id_to_label = {id_to_label}")

        encoded_examples: List[NEREncodedExample] = []
        for idx, raw_example in enumerate(raw_examples):
            raw_example: NERParsedExample = raw_example
            offset_to_label: Dict[int, str] = raw_example.get_offset_label_dict()
            if self.args.prog.local_rank == 0:
                logger.debug(hr())
                logger.debug(f"offset_to_label = {offset_to_label}")
            encoded: BatchEncoding = tokenizer.encode_plus(raw_example.origin,
                                                           max_length=self.args.model.seq_len,
                                                           truncation=TruncationStrategy.LONGEST_FIRST,
                                                           padding=PaddingStrategy.MAX_LENGTH)
            encoded_tokens: List[str] = encoded.tokens()
            if self.args.prog.local_rank == 0:
                logger.debug(hr())
                logger.debug(f"encoded.tokens()           = {encoded.tokens()}")
                for key in encoded.keys():
                    logger.debug(f"encoded[{key:14s}]    = {encoded[key]}")
                logger.debug(hr())
            label_list: List[str] = []
            for token_id in range(self.args.model.seq_len):
                token_repr: str = encoded_tokens[token_id]
                token_span: CharSpan = encoded.token_to_chars(token_id)
                if token_span:
                    token_label = self._decide_span_label(token_span, offset_to_label)
                    label_list.append(token_label)
                    token_sstr = raw_example.origin[token_span.start:token_span.end]
                    if self.args.prog.local_rank == 0:
                        logger.debug('\t'.join(map(str, [token_id, token_repr, token_span, token_sstr, token_label])))
                else:
                    label_list.append('O')
                    if self.args.prog.local_rank == 0:
                        logger.debug('\t'.join(map(str, [token_id, token_repr, token_span])))
            label_ids: List[int] = [label_to_id[label] for label in label_list]
            encoded_example = NEREncodedExample(idx=idx, raw=raw_example, encoded=encoded, label_ids=label_ids)
            encoded_examples.append(encoded_example)
            if self.args.prog.local_rank == 0:
                logger.debug(hr())
                logger.debug(f"label_list                = {label_list}")
                logger.debug(f"label_ids                 = {label_ids}")
                logger.debug(hr())
                logger.debug(f"encoded_example.idx       = {encoded_example.idx}")
                logger.debug(f"encoded_example.raw       = {encoded_example.raw}")
                logger.debug(f"encoded_example.encoded   = {encoded_example.encoded}")
                logger.debug(f"encoded_example.label_ids = {encoded_example.label_ids}")

        if self.args.prog.local_rank == 0:
            logger.info(hr())
            for encoded_example in encoded_examples[:self.args.data.num_check]:
                logger.info("  === [Example %d] ===" % encoded_example.idx)
                logger.info("  = sentence   : %s" % encoded_example.raw.origin)
                logger.info("  = characters : %s" % " | ".join(f"{x}/{y}" for x, y in encoded_example.raw.character_list))
                logger.info("  = tokens     : %s" % " ".join(encoded_example.encoded.tokens()))
                logger.info("  = labels     : %s" % " ".join([id_to_label[x] for x in encoded_example.label_ids]))
                logger.info("  === ")
            logger.info(f"Converted {len(raw_examples)} raw examples to {len(encoded_examples)} encoded examples")
        return encoded_examples

    @staticmethod
    def encoded_examples_to_batch(examples: List[NEREncodedExample]) -> Dict[str, torch.Tensor]:
        first = examples[0]
        batch = {}
        for k, v in first.encoded.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([ex.encoded[k] for ex in examples])
                else:
                    batch[k] = torch.tensor([ex.encoded[k] for ex in examples], dtype=torch.long)
        batch["labels"] = torch.tensor([ex.label_ids for ex in examples],
                                       dtype=torch.long if type(first.label_ids[0]) is int else torch.float)
        batch["example_ids"] = torch.tensor([ex.idx for ex in examples], dtype=torch.int)
        return batch


class NERDataset(Dataset):
    def __init__(self, split: str, tokenizer: PreTrainedTokenizerFast, data: NERCorpus):
        self.data: NERCorpus = data
        examples: List[NERParsedExample] = self.data.read_raw_examples(split)
        self.labels: List[str] = self.data.labels
        self._label_to_id: Dict[str, int] = {label: i for i, label in enumerate(self.labels)}
        self._id_to_label: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}
        self.features: List[NEREncodedExample] = self.data.raw_examples_to_encoded_examples(
            examples, tokenizer, label_list=self.labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> NEREncodedExample:
        return self.features[i]

    def label_to_id(self, label: str) -> int:
        return self._label_to_id[label]

    def id_to_label(self, label_id: int) -> str:
        return self._id_to_label[label_id]


class NERCorpusConverter:
    @classmethod
    def convert_from_kmou_format(cls, infile: str | Path, outfile: str | Path, debug: bool = False):
        with Path(infile).open(encoding="utf-8") as inp, Path(outfile).open("w", encoding="utf-8") as out:
            for line in inp.readlines():
                origin, tagged = line.strip().split("\u241E")
                parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(origin, tagged, debug=debug)
                if parsed:
                    out.write(parsed.to_json(ensure_ascii=False) + "\n")

    @classmethod
    def convert_from_klue_format(cls, infile: str | Path, outfile: str | Path, debug: bool = False):
        with Path(infile) as inp, Path(outfile).open("w", encoding="utf-8") as out:
            raw_text = inp.read_text(encoding="utf-8").strip()
            raw_docs = re.split(r"\n\t?\n", raw_text)
            for raw_doc in raw_docs:
                raw_lines = raw_doc.splitlines()
                num_header = 0
                for line in raw_lines:
                    if not line.startswith("##"):
                        break
                    num_header += 1
                head_lines = raw_lines[:num_header]
                body_lines = raw_lines[num_header:]

                origin = ''.join(x.split("\t")[0] for x in body_lines)
                tagged = head_lines[-1].split("\t")[1].strip()
                parsed: Optional[NERParsedExample] = NERParsedExample.from_tagged(origin, tagged, debug=debug)
                if parsed:
                    character_list_from_head = parsed.character_list
                    character_list_from_body = [tuple(x.split("\t")) for x in body_lines]
                    if character_list_from_head == character_list_from_body:
                        out.write(parsed.to_json(ensure_ascii=False) + "\n")
                    elif debug:
                        print(f"* origin: {origin}")
                        print(f"  tagged: {tagged}")
                        for a, b in zip(character_list_from_head, character_list_from_body):
                            if a != b:
                                print(f"  = {a[0]}:{a[1]} <=> {b[0]}:{b[1]}")
                        print(f"  ====================")
