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
""" XNLI utils (dataset loading and evaluation) """


import logging
import os
import random
import json

from .utils import DataProcessor, InputExample


logger = logging.getLogger(__name__)

def get_metadata(line):
    claimant = line[-3].strip()
    review_date = line[-4].strip()
    language = line[0].strip()
    site = line[1].strip()
    claim_date = line[-5].strip()

    s = 'language : ' + str(language) + ', site : ' + str(site) + ', claimant : ' + str(claimant) + ', claim_date : ' + str(claim_date) + ', review_date: ' + str(review_date)
    return s


class XFactProcessor(DataProcessor):
    """Processor for the XFACT dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, sources=None, data_dir=None, use_metadata=False):
        self.sources = sources
        self.label_sets = {}
        self.label_count = {}
        self.use_metadata = use_metadata
        if data_dir != None and 'chef' not in data_dir:
            for source in self.sources:
                self.get_train_examples(data_dir, source, create_label_set=True)
        elif data_dir != None and 'chef' in data_dir:
            for source in self.sources:
                self.get_chef_train_examples(data_dir, source, create_label_set=True)
        print('Label counts are : ')
        print(self.label_count)
        self.sort_label_set()

    def get_train_examples(self, data_dir, source, create_label_set=False):
        """See base class."""


        print('Loading from file train.{}.tsv'.format(source))
        examples = []
        if source not in self.label_sets:
            self.label_sets[source] = set()
            self.label_count[source] = {}
        lines = self._read_tsv(os.path.join(data_dir, "train.{}.tsv".format(source)))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            if len(line) == 0:
                continue
            claim_text = line[-2]
            label = line[-1].lower()
            metadata = get_metadata(line)
            if create_label_set:
                self.label_sets[source].add(label)
            else:
                if label not in self.label_sets[source]:
                    raise ValueError('Label {} not found in set {}'.format(label, str(self.label_sets[source])))
            if label not in self.label_count[source]:
                self.label_count[source][label] = 0
            self.label_count[source][label] +=1
            assert isinstance(claim_text, str) and  isinstance(label, str)
            if self.use_metadata:
                claim_text = claim_text + ' ' + metadata
            examples.append(InputExample(guid=guid, text_a=claim_text, text_b=None, label=label))
        print('Examples loaded from source {} : {}'.format(source, len(examples)))

        random.shuffle(examples)
        return examples[:200]
        # return examples

    def get_chef_train_examples(self, data_dir, source, create_label_set=False):
        print('Loading CHEF from file {}'.format(data_dir))
        id2label = {
            0: 'true',
            1: 'false',
            2: 'complicated/hard to categorise'
        }
        examples = []
        if source not in self.label_sets:
            self.label_sets[source] = set()
            self.label_count[source] = {}
        lines = json.load(open(os.path.join(data_dir, "CHEF_train.json"), 'r', encoding='utf-8'))
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            claim_text = line['claim']
            label = id2label[line['label']]
            if create_label_set:
                self.label_sets[source].add(label)
            else:
                if label not in self.label_sets[source]:
                    raise ValueError('Label {} not found in set {}'.format(label, str(self.label_sets[source])))
            if label not in self.label_count[source]:
                self.label_count[source][label] = 0
            self.label_count[source][label] +=1
            assert isinstance(claim_text, str) and  isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=claim_text, text_b=None, label=label))
        print('Examples loaded from source {} : {}'.format(source, len(examples)))

        random.shuffle(examples)
        return examples

    def get_chef_dev_examples(self, data_dir, source, filename=None):
        """See base class."""
        id2label = {
            0: 'true',
            1: 'false',
            2: 'complicated/hard to categorise'
        }
        examples = []
        examples_skipped = 0
        lines = json.load(open(os.path.join(data_dir, "CHEF_test.json"), 'r', encoding='utf-8'))
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            claim_text = line['claim']
            label = id2label[line['label']]
            if label not in self.label_sets[source]:
                examples_skipped +=1
                continue
            assert isinstance(claim_text, str) and  isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=claim_text, text_b=None, label=label))
        return examples

    def get_dev_examples(self, data_dir, source, filename=None):
        """See base class."""

        examples = []
        examples_skipped = 0
        if filename is None:
            lines = self._read_tsv(os.path.join(data_dir, "dev.{}.tsv".format(source)))
        else:
            lines = self._read_tsv(os.path.join(data_dir, filename))
        #lines = self._read_tsv(os.path.join(data_dir, "train.{}.tsv".format(source)))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            claim_text = line[-2]
            label = line[-1].lower()
            metadata = get_metadata(line)
            if label not in self.label_sets[source]:
                #print('Skipping test example')
                examples_skipped +=1
                continue
            assert isinstance(claim_text, str) and  isinstance(label, str)
            if self.use_metadata:
                claim_text = claim_text + ' ' + metadata
            examples.append(InputExample(guid=guid, text_a=claim_text, text_b=None, label=label))
        #print('For source: {}, dev examples skipped: {}, Examples left: {}'.format(source, examples_skipped, len(examples)))
        return examples

    def sort_label_set(self):
        sorted_sets = {}

        for key, val in self.label_sets.items():
            sorted_sets[key] = sorted(list(val))

        self.label_sets = sorted_sets

    def get_test_examples(self, data_dir, source):
        """See base class."""

        examples = []
        examples_skipped = 0
        lines = self._read_tsv(os.path.join(data_dir, "test.{}.tsv".format(source)))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            claim_text = line[-2]
            label = line[-1].lower()
            metadata = get_metadata(line)
            if label not in self.label_sets[source]:
                #print('Skipping test example')
                examples_skipped +=1
                continue
            assert isinstance(claim_text, str) and  isinstance(label, str)
            if self.use_metadata:
                claim_text = claim_text + ' ' + metadata
            examples.append(InputExample(guid=guid, text_a=claim_text, text_b=None, label=label))
        print('For source: {}, test examples skipped: {}, Examples left: {}'.format(source, examples_skipped, len(examples)))
        return examples

    def get_labels(self):
        """See base class."""
        self.sort_label_set()
        return self.label_sets


xfact_processors = {
    "xfact": XFactProcessor,
}

xfact_output_modes = {
    "xfact": "classification",
}

#xnli_tasks_num_labels = {
#    "xnli": 3,
#}
