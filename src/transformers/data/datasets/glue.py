# -*- coding: utf-8 -*-
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm, trange

from ...tokenization_bart import BartTokenizer, BartTokenizerFast
from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, dssm_convert_examples_to_features, dssm_convert_examples_a_to_features, dssm_convert_examples_with_neg_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures, InputExample, InputExampleWithNeg, InputFeaturesDssm, InputFeaturesWithNegDssm


logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_size: int = field(
        default=0,
        metadata={
            "help": "lines of data, if data_size is set to 0, will scan the files and get data lines"
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    dev_label = "dev_label"
    dev_a = "dev_a"
    dev_b = "dev_b"


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

class DssmDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeaturesDssm]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        tokenizer_b: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()

        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.dev_label:
                    examples = self.processor.get_dev_label_examples(args.data_dir)
                    print('dev label example: ', examples[0])
                elif mode == Split.dev_a:
                    examples = self.processor.get_dev_a_examples(args.data_dir)
                    print('dev a example: ', examples[0])
                elif mode == Split.dev_b:
                    examples = self.processor.get_dev_b_examples(args.data_dir)
                    # print('dev b example: ', examples[0])
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.guids = [example.guid for example in examples]
                
                # if mode == Split.dev_label:
                #     pass
                if mode == Split.dev_a or mode == Split.dev_b or mode == Split.dev_label:
                    self.features = dssm_convert_examples_a_to_features(
                        examples,
                        tokenizer,
                        max_length=args.max_seq_length,
                        label_list=label_list,
                        output_mode=self.output_mode,
                    )
                
                else:
                    self.features = dssm_convert_examples_to_features(
                        examples,
                        tokenizer,
                        tokenizer_b,
                        max_length=args.max_seq_length,
                        max_length_b=args.max_seq_length,
                        label_list=label_list,
                        output_mode=self.output_mode,
                    )
                start = time.time()
                # torch.save(self.features, cached_features_file)
                # # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                # logger.info(
                #     "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                # )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesDssm:
        return self.features[i]

    def get_labels(self):
        return self.label_list

class DssmDatasetDisk(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    fileIndex = 0
    output_mode: str
    features: List[InputFeaturesDssm]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        tokenizer_b: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        limit_length_b: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        number = args.data_size
        if number==0:
            with open(os.path.join(args.data_dir, "train.tsv"), "r", encoding="utf-8-sig") as f:
                for _ in tqdm(f,desc="load training dataset"):
                    number+=1
        self.number = number
        self.fopen = open(os.path.join(args.data_dir, "train.tsv"), "r", encoding="utf-8-sig")
        self.tokenizer = tokenizer
        self.tokenizer_b = tokenizer_b
        self.fopen.__next__()

    def __len__(self):
        return self.number

    def __getitem__(self, i) -> InputFeaturesDssm:
        self.fileIndex += 1
        line = self.fopen.__next__()
        items = line.strip().split("\t")

        examples = []
        guid = "%s-%s" % ("train", self.fileIndex)
        text_a = items[3]
        text_b = items[4]

        label =  items[0]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        features = dssm_convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    self.tokenizer_b,
                    max_length=self.args.max_seq_length,
                    max_length_b=self.args.max_seq_length,
                    label_list=self.label_list,
                    output_mode=self.output_mode,
                )

        # label = self.label_map[items[0]]
        # batch_encoding = self.tokenizer(
        #     [(text_a, text_b)],
        #     max_length=self.args.max_seq_length,
        #     #padding=True,
        #     padding="max_length",
        #     truncation=True,
        # )
        # modelInput = {k: batch_encoding[k][0] for k in batch_encoding}
        # feature = InputFeatures(**modelInput, label=label)
        # logger.info("features: %s" % features)

        return features[0]

    def get_labels(self):
        return self.label_list

class DssmDatasetDiskForRanking(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    fileIndex = 0
    output_mode: str
    features: List[InputFeaturesWithNegDssm]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        tokenizer_b: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        limit_length_b: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        number = args.data_size
        if number==0:
            with open(os.path.join(args.data_dir, "train.tsv"), "r", encoding="utf-8-sig") as f:
                for _ in tqdm(f,desc="load training dataset"):
                    number+=1
        self.number = number
        self.fopen = open(os.path.join(args.data_dir, "train.tsv"), "r", encoding="utf-8-sig")
        self.tokenizer = tokenizer
        self.tokenizer_b = tokenizer_b
        self.fopen.__next__()

    def __len__(self):
        return self.number

    def __getitem__(self, i) -> InputFeaturesWithNegDssm:
        self.fileIndex += 1
        line = self.fopen.__next__()
        items = line.strip().split("\t")

        examples = []
        guid = "%s-%s" % ("train", self.fileIndex)
        text_a = items[5]
        text_b = items[6]
        text_c = items[7]
        text_d = items[8]

        label =  items[0]

        # print(text_a)
        # print(text_b)
        # print(text_c)
        # print(text_d)
        examples.append(InputExampleWithNeg(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        features = dssm_convert_examples_with_neg_to_features(
                    examples,
                    self.tokenizer,
                    self.tokenizer_b,
                    max_length=self.args.max_seq_length,
                    max_length_b=self.args.max_seq_length,
                    label_list=self.label_list,
                    output_mode=self.output_mode,
                )

        # label = self.label_map[items[0]]
        # batch_encoding = self.tokenizer(
        #     [(text_a, text_b)],
        #     max_length=self.args.max_seq_length,
        #     #padding=True,
        #     padding="max_length",
        #     truncation=True,
        # )
        # modelInput = {k: batch_encoding[k][0] for k in batch_encoding}
        # feature = InputFeatures(**modelInput, label=label)
        # logger.info("features: %s" % features)

        return features[0]

    def get_labels(self):
        return self.label_list