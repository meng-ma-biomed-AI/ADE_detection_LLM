# coding=utf-8
# Copyright 2020- The Google AI Language Team Authors and The HuggingFace Inc. team and Facebook Inc.
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
"""processors and helpers for classification"""

import csv
import logging
import os

import numpy as np

from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        elif output_mode == 'multilabel_classification':
            label = [label_map[l] for l in example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if output_mode == 'multilabel_classification':
                logger.info("label: %s (id = %s)" % (example.label, str(label)))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


def _read_csv(input_file, quotechar='"'):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter=",", quotechar=quotechar))


class N2c2AdeProcessor(DataProcessor):
    """Processor for the adverse drug event relations preprocessed dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = str(i)
                text_a = line[0]
                label = line[1]
                if '@ADE$' not in text_a:
                    continue
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfAdeProcessor(DataProcessor):
    """Processor for the serious adverse event relations preprocessed dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_ade_triples_nearby_nomedsyns_us_0.8.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade_triples_nearby_nomedsyns.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade_triples_nearby_nomedsyns.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfUndersampled50AdeProcessor(DataProcessor):
    """Processor for the serious adverse event relations undersampled dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_undersampled_0.5.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfUndersampled70AdeProcessor(DataProcessor):
    """Processor for the serious adverse event relations undersampled dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_undersampled_0.7.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfUndersampled80AdeProcessor(DataProcessor):
    """Processor for the serious adverse event relations undersampled dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_undersampled_0.8.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfUndersampled90AdeProcessor(DataProcessor):
    """Processor for the serious adverse event relations undersampled dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_undersampled_0.9.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class UcsfIsAdeInNoteProcessor(DataProcessor):
    """Processor for the serious adverse event relations undersampled dataset"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_ade_in_note.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_ade_in_note.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_ade_in_note.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class UcsfMedBeforeHospProcessor(DataProcessor):
    """Processor for the relations between med and hosp events: 1 indicates that the patient was on med before hosp"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_med_before_hosp_us_0.8.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_med_before_hosp.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_med_before_hosp.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class UcsfHospForAeProcessor(DataProcessor):
    """Processor for the relations between med and hosp events: 1 indicates that the patient was on med before hosp"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "train_hosp_for_ae_us_0.8.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "dev_hosp_for_ae.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(_read_csv(os.path.join(data_dir, "test_hosp_for_ae.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']  # 0: no relation, 1: there exists a relation

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = line[1]
                text_a = line[2]
                label = line[3]

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

processors = {
    "ade": N2c2AdeProcessor,
    "ucsf_ade": UcsfAdeProcessor,
    "ucsf_ade_undersampled_0.5": UcsfUndersampled50AdeProcessor,
    "ucsf_ade_undersampled_0.7": UcsfUndersampled70AdeProcessor,
    "ucsf_ade_undersampled_0.8": UcsfUndersampled80AdeProcessor,
    "ucsf_ade_undersampled_0.9": UcsfUndersampled90AdeProcessor,
    "ucsf_ade_hierarchical": UcsfAdeProcessor,
    "ucsf_is_ade_in_note": UcsfIsAdeInNoteProcessor,
    "ucsf_med_before_hosp": UcsfMedBeforeHospProcessor,
    "ucsf_hosp_for_ae": UcsfHospForAeProcessor,
}


output_modes = {
    "ade": "classification",
    "ucsf_ade": "classification",
    "ucsf_ade_undersampled_0.5": "classification",
    "ucsf_ade_undersampled_0.7": "classification",
    "ucsf_ade_undersampled_0.8": "classification",
    "ucsf_ade_undersampled_0.9": "classification",
    "ucsf_ade_hierarchical": "classification",
    "ucsf_is_ade_in_note": "classification",
    "ucsf_med_before_hosp": "classification",
    "ucsf_hosp_for_ae": "classification",
}

stopping_metrics = {
    "ade": "micro_f1",
    "ucsf_ade": "f1",
    "ucsf_ade_undersampled_0.5": "f1",
    "ucsf_ade_undersampled_0.7": "f1",
    "ucsf_ade_undersampled_0.8": "f1",
    "ucsf_ade_undersampled_0.9": "f1",
    "ucsf_ade_hierarchical": "f1",
    "ucsf_is_ade_in_note": "f1",
    "ucsf_med_before_hosp": "f1",
    "ucsf_hosp_for_ae": "f1",
}


def multiclass_acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, )
    prec = precision_score(y_true=labels, y_pred=preds, )
    recall = recall_score(y_true=labels, y_pred=preds, )
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    macro_precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    macro_recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": recall,
        'micro_f1': micro_f1,
        "macro_f1": macro_f1,
        "macro_weighted_f1": macro_weighted_f1,
        "macro_precision": macro_precision,
        "macro_weighted_precision": macro_weighted_precision,
        "macro_recall": macro_recall,
        "macro_weighted_recall": macro_weighted_recall,
        "confusion_matrix": confusion,
    }


def get_precision_recall_curve(probs, labels, pos_label=1):
    pos_probs = probs[:, pos_label]
    precisions, recalls, thresholds = precision_recall_curve(probas_pred=pos_probs, y_true=labels)
    return {
        "thresholded_precisions": precisions,
        "thresholded_recalls": recalls,
        "thresholds": thresholds,
    }


def get_roc_curve(probs, labels, pos_label=1):
    pos_probs = probs[:, pos_label]
    fpr, tpr, thresholds = roc_curve(y_score=pos_probs, y_true=labels)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def acc_and_micro_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "micro_f1": micro_f1,
    }


def acc_p_r_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, )
    recall = recall_score(y_true=labels, y_pred=preds, )
    precision = precision_score(y_true=labels, y_pred=preds, )

    return {
        "acc": acc,
        "f1": f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics(task_name, preds, labels, examples, probs=None):
    assert len(preds) == len(labels) == len(examples)
    if task_name in ['ucsf_ade',
                     'ucsf_ade_undersampled_0.5',
                     'ucsf_ade_undersampled_0.7',
                     'ucsf_ade_undersampled_0.8',
                     'ucsf_ade_undersampled_0.9',
                     'ucsf_ade_hierarchical',
                     'ucsf_is_ade_in_note',
                     'ucsf_med_before_hosp',
                     'ucsf_hosp_for_ae',
                     ]:
        results = multiclass_acc_and_f1(preds, labels)
        if probs is not None:
            results.update(get_roc_curve(probs, labels))
        return results
    elif task_name == "ade":
        return multiclass_acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
