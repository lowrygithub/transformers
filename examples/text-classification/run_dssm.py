# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""




import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceEmbedding, AutoTokenizer, EvalPrediction, GlueDataset, DssmDataset, DssmDatasetDisk, DssmDatasetDiskForRanking
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_b: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    circle_loss_m: float = field(
        default=0.25,
        metadata={
            "help": "circle_loss_m "
        },
    )
    circle_loss_gamma: float = field(
        default=2,
        metadata={
            "help": "circle_loss_gamma "
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_a = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config_b = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path_b,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer_a = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path_b,
        cache_dir=model_args.cache_dir,
    )
    model_a = AutoModelForSequenceEmbedding.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config_a,
        cache_dir=model_args.cache_dir,
    )
    model_b = AutoModelForSequenceEmbedding.from_pretrained(
        model_args.model_name_or_path_b,
        from_tf=bool(".ckpt" in model_args.model_name_or_path_b),
        config=config_b,
        cache_dir=model_args.cache_dir,
    )
    # Get datasets

    if training_args.loss_function == "triplet_hard":
        train_dataset = (
            DssmDatasetDiskForRanking(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, cache_dir=model_args.cache_dir) if training_args.do_train else None
        )
    else:
        train_dataset = (
            DssmDatasetDisk(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, cache_dir=model_args.cache_dir) if training_args.do_train else None
        )
    
    eval_dataset_a = (
        DssmDataset(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, mode="dev_a", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    # print("eval_dataset_a.guids:", eval_dataset_a.guids[:10])

    eval_dataset_b = (
        DssmDataset(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, mode="dev_b", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    # print("eval_dataset_b.guids:", eval_dataset_b.guids[:10])

    eval_dataset_label = (
        DssmDataset(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, mode="dev_label", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    # print("eval_dataset_label.guids:", eval_dataset_label.guids[:10])

    test_dataset = (
        DssmDataset(data_args, tokenizer=tokenizer_a, tokenizer_b=tokenizer_b, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )
    # # Get datasets
    # train_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer_a, cache_dir=model_args.cache_dir) if training_args.do_train else None
    # )
    # eval_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer_a, mode="dev", cache_dir=model_args.cache_dir)
    #     if training_args.do_eval
    #     else None
    # )
    # test_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer_a, mode="test", cache_dir=model_args.cache_dir)
    #     if training_args.do_predict
    #     else None
    # )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model_a,
        model_b=model_b,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        eval_dataset_label=eval_dataset_label,
        eval_dataset_a=eval_dataset_a,
        eval_dataset_b=eval_dataset_b,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer_a.save_pretrained(training_args.output_dir)

    # Evaluation
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        
        # print('eval_dataset_label:', eval_dataset_label)
        # print('eval_dataset_a:', eval_dataset_a)
        # print('eval_dataset_b:', eval_dataset_b)
       

        
        trainer.compute_metrics = build_compute_metrics_fn(eval_dataset_a.args.task_name)
        eval_result = trainer.evaluate(eval_dataset_label=eval_dataset_label, eval_dataset_a=eval_dataset_a, eval_dataset_b=eval_dataset_b)

        output_eval_file = os.path.join(
            model_args.model_name_or_path, f"eval_results_{eval_dataset_a.args.task_name}.txt"
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset_a.args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer_a, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
