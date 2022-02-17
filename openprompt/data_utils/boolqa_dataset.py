from argparse import RawDescriptionHelpFormatter
from doctest import Example
from email.errors import InvalidMultipartContentTransferEncodingDefect
from turtle import pos

from scipy import rand
from torch import positive
from openprompt.data_utils.utils import InputExample
import os
import json, csv, re
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Optional
from tqdm import tqdm

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor

import numpy as np
import random


class DuReaderBoolQADataset(DataProcessor):

    def __init__(self):
        super().__init__(["否", "是"])
        self.YESNO_LABELS = ['Yes', 'No', 'Depends']
        self.LABELS_MAP = {
            "Yes": "是",
            "No": "否",
        #     "Depends": "",
        }
        self.CONTEXT_LENGTH = [50, 800]
    
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
       
        examples = []
        raw_data = []

        path = os.path.join(data_dir, "{}.json".format(split))
        with open(path, encoding="UTF-8") as load_f:
            for line in load_f:
                decoder = json.JSONDecoder()
                raw_data.append(decoder.decode(line))

        count_yes = 0
        count_no = 0
        count_depend = 0
        count_global = 0
        count_to_long = 0
        count_to_short = 0

        for i in tqdm(range(len(raw_data)), desc="Processing {} data of DuReaderBoolQA".format(split)):

            sample = raw_data[i]
            
            if sample["yesno_answer"] == self.YESNO_LABELS[2]:
                count_depend += 1
                continue

            question = "".join(sample["question"]).replace(" ", "").replace("，", ",").replace("。", ".")
            label_text = self.LABELS_MAP[sample["yesno_answer"]]
            count = 0
            label = 0

            if sample["yesno_answer"] == self.YESNO_LABELS[1]:
                label = 1
            
            for context in sample["documents"]:
                
                context = "".join(context["paragraphs"]).replace(" ", "").replace("，", ",").replace("。", ".")

                if len(context) > self.CONTEXT_LENGTH[1]:
                    count_to_long += 1
                    continue
                elif len(context) < self.CONTEXT_LENGTH[0]:
                    count_to_short += 1
                    continue
                
                new_data = InputExample(
                    guid=str(count_global),
                    text_a=context,
                    text_b=question,
                    tgt_text=label_text,
                    label=label,

                )
                
                examples.append(new_data)
                count += 1
                count_global += 1
            
            if label == 1:
                count_yes += count
            else:
                count_no += count

        print("Total:%d, Yes:%d, No:%d" % (count_global, count_yes, count_no))
        print("Deleted:\n\tTo_long:%d, To_short:%d" % (count_to_long, count_to_short))

        return examples


class IFLYTEKBoolQADataset(DataProcessor):

    def __init__(self):
        super().__init__(["否", "是"])
        self.CONTEXT_LENGTH = [50, 800]
        self.id2topic = None

    
    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
        
        # load topics
        if self.id2topic == None:
            self.load_topics(data_dir)
        
        examples = []
        raw_data = []

        path = os.path.join(data_dir, "{}.json".format(split))
        with open(path, encoding="UTF-8") as load_f:
            decoder = json.JSONDecoder()
            for line in load_f:
                raw_data.append(decoder.decode(line))

        if split == "train":
            examples = self.get_lite_examples(raw_data=raw_data, split=split, contrast=True)
        elif split == "dev":
            examples = self.get_lite_examples(raw_data=raw_data, split=split, contrast=True)
        elif split == "test":
            examples = self.get_lite_examples(raw_data=raw_data, split=split, contrast=True)

        return examples
        
    def get_lite_examples(self, raw_data: Optional[list] = None, split: Optional[str] = None, contrast: bool = False) -> List[InputExample]:

        examples = []

        count_yes = 0
        count_no = 0
        count_global = 0
        count_to_long = 0
        count_to_short = 0

        for i in tqdm(range(len(raw_data)), desc="Processing {} data of IFLYTEKBoolQA".format(split)):

            sample = raw_data[i]

            correct_topic = int(sample["label"], base=10)
            correct_topic_text = sample["label_des"]

            # delete the topic 'others'
            if correct_topic == 118:
                continue

            context = "".join(sample["sentence"]).replace(" ", "").replace("，", ",").replace("。", ".")
            if len(context) > self.CONTEXT_LENGTH[1]:
                count_to_long += 1
                continue
            elif len(context) < self.CONTEXT_LENGTH[0]:
                count_to_short += 1
                continue

            positive_data = InputExample(
                guid=str(count_global),
                text_a=context,
                text_b=correct_topic_text,
                tgt_text=self.labels[1],
                label=1,
            )
            examples.append(positive_data)
            count_global += 1
            count_yes += 1

            if contrast:
                incorrect_topic = random.randint(0, 117)
                if incorrect_topic >= correct_topic:
                    incorrect_topic += 1
                incorrect_topic_text = self.id2topic[incorrect_topic]

                negative_data = InputExample(
                    guid=str(count_global),
                    text_a=context,
                    text_b=incorrect_topic_text,
                    tgt_text=self.labels[0],
                    label=0,
                )
                examples.append(negative_data)
                count_global += 1
                count_no += 1

        print("Total:%d, Yes:%d, No:%d" % (count_global, count_yes, count_no))
        print("Deleted:\n\tTo_long:%d, To_short:%d" % (count_to_long, count_to_short))

        print(examples[0].text_a)
        print(examples[0].text_b)

        return examples

    def get_full_examples(self, raw_data: Optional[list] = None, split: Optional[str] = None) -> List[InputExample]:

        examples = []

        count_yes = 0
        count_no = 0
        count_global = 0
        count_to_long = 0
        count_to_short = 0

        for i in tqdm(range(len(raw_data)), desc="Processing {} data of IFLYTEKBoolQA".format(split)):

            sample = raw_data[i]

            correct_topic = int(sample["label"], base=10)

            # delete the topic 'others'
            if correct_topic == 118:
                continue

            context = "".join(sample["sentence"]).replace(" ", "").replace("，", ",").replace("。", ".")
            if len(context) > self.CONTEXT_LENGTH[1]:
                count_to_long += 1
                continue
            elif len(context) < self.CONTEXT_LENGTH[0]:
                count_to_short += 1
                continue

            for topic, topic_text in self.id2topic.items():

                topic_text = self.id2topic[topic]
                label = 0

                if topic == correct_topic:
                    label = 1
                    count_yes += 1
                else:
                    count_no +=1

                new_example = InputExample(
                    guid=str(count_global),
                    text_a=context,
                    text_b=topic_text,
                    tgt_text=self.labels[label],
                    label=label,
                )
                examples.append(new_example)
                count_global += 1

        print("Total:%d, Yes:%d, No:%d" % (count_global, count_yes, count_no))
        print("Deleted:\n\tTo_long:%d, To_short:%d" % (count_to_long, count_to_short))

        print(examples[0].text_a)
        print(examples[0].text_b)

        return examples

    def get_random_example(self, raw_data: Optional[list] = None, split: Optional[str] = None) -> List[InputExample]:
        
        examples = []

        count_yes = 0
        count_no = 0
        count_global = 0
        count_to_long = 0
        count_to_short = 0

        for i in tqdm(range(len(raw_data)), desc="Processing {} data of IFLYTEKBoolQA".format(split)):

            sample = raw_data[i]

            correct_topic = int(sample["label"], base=10)
            label = 0

            # delete the topic 'others'
            if correct_topic == 118:
                continue

            context = "".join(sample["sentence"]).replace(" ", "").replace("，", ",").replace("。", ".")
            if len(context) > self.CONTEXT_LENGTH[1]:
                count_to_long += 1
                continue
            elif len(context) < self.CONTEXT_LENGTH[0]:
                count_to_short += 1
                continue

            random_topic = random.randint(0, 118)
            if random_topic == correct_topic:
                label == 1
                count_yes += 1
            else:
                count_no += 1
            topic_text = self.id2topic[random_topic]

            random_data = InputExample(
                guid=str(count_global),
                text_a=context,
                text_b=topic_text,
                tgt_text=self.labels[label],
                label=label,
            )
            examples.append(random_data)
            count_global += 1

        print("Total:%d, Yes:%d, No:%d" % (count_global, count_yes, count_no))
        print("Deleted:\n\tTo_long:%d, To_short:%d" % (count_to_long, count_to_short))

        print(examples[0].text_a)
        print(examples[0].text_b)

        return examples

    def load_topics(self, data_dir: Optional[str] = None) -> None:

        self.id2topic = {}
        count_topic = 0
        topics_path = os.path.join(data_dir, "{}.json".format("labels"))
        with open(topics_path, encoding="UTF-8") as load_f:
            decoder = json.JSONDecoder()
            for line in load_f:
                topic = decoder.decode(line)
                self.id2topic[int(topic["label"], base=10)] = topic["label_des"]
                count_topic += 1

        print("Load topics:%d" % (count_topic))

        return


PROCESSORS = {
    "dureaderboolqa": DuReaderBoolQADataset,
    "iflytekboolqa": IFLYTEKBoolQADataset,
}
