from argparse import RawDescriptionHelpFormatter
from doctest import Example
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


YESNO_LABELS = ['Yes', 'No', 'Depends']
LABELS_MAP = {
    "Yes": "是",
    "No": "否",
#     "Depends": "",
}
LABELS = ["否", "是"]
CONTEXT_LENGTH = [50, 800]


class DuReaderBoolQADataset(DataProcessor):

    def __init__(self):
        super().__init__(LABELS)
    
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
            
            if sample["yesno_answer"] == YESNO_LABELS[2]:
                count_depend += 1
                continue

            question = "".join(sample["question"]).replace(" ", "").replace("，", ",").replace("。", ".")
            label_text = LABELS_MAP[sample["yesno_answer"]]
            count = 0
            label = 0

            if sample["yesno_answer"] == YESNO_LABELS[1]:
                label = 1
            
            for context in sample["documents"]:
                
                # new_data = {
                #     "id": count_global,
                #     "label": label,
                #     "question": question,
                #     # "title": context["title"],
                #     "context": "".join(context["paragraphs"]).replace(" ", "").replace("，", ",").replace("。", "."),
                # }
                context = "".join(context["paragraphs"]).replace(" ", "").replace("，", ",").replace("。", ".")
                if len(context) > CONTEXT_LENGTH[1]:
                    count_to_long += 1
                    continue
                elif len(context) < CONTEXT_LENGTH[0]:
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


PROCESSORS = {
    "dureaderboolqa": DuReaderBoolQADataset,
}
