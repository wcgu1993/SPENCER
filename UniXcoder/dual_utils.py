from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, code_token, summary_token=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.code_token = code_token
        self.summary_token = summary_token

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, code_input, code_mask, summary_input, summary_mask):
        self.code_input = code_input
        self.code_mask = code_mask
        self.summary_input = summary_input
        self.summary_mask = summary_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 2:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            code_token = line[0]
            summary_token = line[1]
            examples.append(
                InputExample(guid=guid, code_token=code_token, summary_token=summary_token))
        return examples


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """


    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        code_token = tokenizer.tokenize(example.code_token)[:50]

        summary_token = tokenizer.tokenize(example.summary_token)[:50]
        # Account for [CLS] and [SEP] with "- 2"
        if len(code_token) > max_seq_length - 2:
            code_token = code_token[:(max_seq_length - 2)]
        if len(summary_token) > max_seq_length - 2:
            summary_token = summary_token[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        code_token = code_token + [sep_token]

        summary_token = summary_token + [sep_token]

        if cls_token_at_end:
            code_token = code_token + [cls_token]
            summary_token = summary_token + [cls_token]
        else:
            code_token = [cls_token] + code_token
            summary_token = [cls_token] + summary_token

        code_input = tokenizer.convert_tokens_to_ids(code_token)
        summary_input = tokenizer.convert_tokens_to_ids(summary_token)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        code_mask = [1 if mask_padding_with_zero else 0] * len(code_input)
        summary_mask = [1 if mask_padding_with_zero else 0] * len(summary_input)

        # Zero-pad up to the sequence length.
        code_padding_length = max_seq_length - len(code_input)
        summary_padding_length = max_seq_length - len(summary_input)
        if pad_on_left:
            code_input = ([pad_token] * code_padding_length) + code_input
            code_mask = ([0 if mask_padding_with_zero else 1] * code_padding_length) + code_mask
            summary_input = ([pad_token] * summary_padding_length) + summary_input
            summary_mask = ([0 if mask_padding_with_zero else 1] * summary_padding_length) + summary_mask
            
        else:
            code_input = code_input + ([pad_token] * code_padding_length)
            code_mask = code_mask + ([0 if mask_padding_with_zero else 1] * code_padding_length)
            summary_input = summary_input + ([pad_token] * summary_padding_length)
            summary_mask = summary_mask + ([0 if mask_padding_with_zero else 1] * summary_padding_length)

        assert len(code_input) == max_seq_length
        assert len(code_mask) == max_seq_length
        assert len(summary_input) == max_seq_length
        assert len(summary_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("code_tokens: %s" % " ".join(
                [str(x) for x in code_token]))
            logger.info("summary_tokens: %s" % " ".join(
                [str(x) for x in summary_token]))
            logger.info("code_input: %s" % " ".join([str(x) for x in code_input]))
            logger.info("code_mask: %s" % " ".join([str(x) for x in code_mask]))
            logger.info("summary_input: %s" % " ".join([str(x) for x in summary_input]))
            logger.info("summary_mask: %s" % " ".join([str(x) for x in summary_mask]))

        features.append(
            InputFeatures(code_input=code_input,
                          code_mask=code_mask,
                          summary_input=summary_input,
                          summary_mask=summary_mask))
    return features

processors = {
    "codesearch": CodesearchProcessor,
}