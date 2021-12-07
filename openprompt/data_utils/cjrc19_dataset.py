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


class CJRCProcessor(DataProcessor):
    # TODO Implement needed

    def __init__(self, is_bool: bool = False):
        super().__init__()
        self.is_bool = is_bool

    def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
        
        if self.is_bool:
            return self.get_bool_examples(data_dir=data_dir, split=split)

        examples = []
        path = os.path.join(data_dir, "{}.json".format(split))

        with open(path, encoding="UTF-8") as f:
            raw_data_dict = json.load(f)

        raw_data_dict = raw_data_dict["data"]
        num_example = len(raw_data_dict)

        # Q: 是否需要保留domain、case_name信息

        context_lst = [sample["paragraphs"][0]["context"] for sample in raw_data_dict]
        qas_lst = [sample["paragraphs"][0]["qas"] for sample in raw_data_dict]

        # pack each context, question, answer

        guid = 0
        impossible = 0

        for i in tqdm(range(num_example), desc="Processing {} data of CJRC_2019".format(split)):

            for qas in qas_lst[i]:

                if qas["is_impossible"] == "true":

                    example = InputExample(
                        guid=str(guid),
                        text_a=context_lst[i],
                        text_b=qas["question"],
                        tgt_text="不知道")

                    impossible += 1

                else:
                    example = InputExample(
                        guid=str(guid),
                        text_a=context_lst[i],
                        text_b=qas["question"],
                        tgt_text=qas["answers"][0]["text"])

                examples.append(example)
                guid += 1

        print("Finished. Got {} {} examples, including {} unanswerable questions.".format(str(guid), split, impossible))

        return examples

    def get_bool_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
        # TODO Not implemented
        examples = []
        path = os.path.join(data_dir, "{}.json".format(split))

        with open(path, encoding="UTF-8") as f:
            raw_data_dict = json.load(f)

        raw_data_dict = raw_data_dict["data"]
        num_example = len(raw_data_dict)

        # Q: 是否需要保留domain、case_name信息

        context_lst = [sample["paragraphs"][0]["context"] for sample in raw_data_dict]
        qas_lst = [sample["paragraphs"][0]["qas"] for sample in raw_data_dict]

        # pack each context, question, answer

        guid = 0
        deleted_impossible_qas = 0
        deleted_complicated_qas = 0
        yes_qas = 0
        no_qas = 0
        deleted_case = 0

        
        # filtering
        for i in tqdm(range(num_example), desc="Processing {} data of CJRC_2019".format(split)):

            case_is_deleted = True

            for qas in qas_lst[i]:
                
                if qas["is_impossible"] == "true":
                    deleted_impossible_qas += 1

                elif "是否" not in qas["question"]:        
                    deleted_complicated_qas += 1

                else:
                    
                    ans = ""
                    label = 1
                    if qas["answers"][0]["text"] == "YES":
                        ans += "是"
                        yes_qas += 1
                        label = 1
                    elif qas["answers"][0]["text"] == "NO":
                        ans += "否"
                        no_qas += 1
                        label = 0
                    else:
                        deleted_impossible_qas += 1
                        continue

                    example = InputExample(
                        guid=str(guid),
                        text_a=context_lst[i],
                        text_b=qas["question"],
                        label=label,
                        tgt_text=ans)
                    
                    examples.append(example)
                    guid += 1
                    case_is_deleted = False
            
            if case_is_deleted:
                deleted_case += 1

        print("Finished:\n\tGot {} {} examples, including {} true qas ans {} false qas.\n\tDelete {} unanswerable questions, {} complicated questions.".format(str(guid), split, yes_qas, no_qas, deleted_impossible_qas, deleted_complicated_qas))

        return examples


# class CJRCMLMProcessor(DataProcessor):

    # def __init__(self):
    #     super().__init__()
    #     self.labels = None

    # def get_examples(self, data_dir: Optional[str] = None, split: Optional[str] = None) -> List[InputExample]:
    #     # TODO Not implemented
    #     examples = []
    #     path = os.path.join(data_dir, "{}.json".format(split))

    #     with open(path, encoding="UTF-8") as f:
    #         raw_data_dict = json.load(f)

    #     raw_data_dict = raw_data_dict["data"]
    #     num_example = len(raw_data_dict)

    #     # Q: 是否需要保留domain、case_name信息

    #     context_lst = [sample["paragraphs"][0]["context"] for sample in raw_data_dict]
    #     qas_lst = [sample["paragraphs"][0]["qas"] for sample in raw_data_dict]

    #     # pack each context, question, answer

    #     guid = 0
    #     deleted_impossible_qas = 0
    #     deleted_complicated_qas = 0
    #     deleted_not_num_qas = 0
    #     deleted_case = 0

    #     p = r'^[^\d]*\d+(\.\d)+?[^\d\.]*$'
    #     p_ans = r'^[^\d]*([\d\.]+)[^\d]*$'
    #     p_before = r'^([^\d]*)[\d\.]+[^\d]*$'
    #     p_after = r'^[^\d]*[\d\.]+([^\d]*)$'

        
    #     # filtering
    #     for i in tqdm(range(num_example), desc="Processing {} data of CJRC_2019".format(split)):

    #         case_is_deleted = True

    #         for qas in qas_lst[i]:
                
    #             if qas["is_impossible"] == "true":
    #                 deleted_impossible_qas += 1

    #             elif "多少" not in qas["question"]:        
    #                 deleted_complicated_qas += 1

    #             elif re.match(p, qas["answers"][0]["text"]):
    #                 deleted_not_num_qas += 1

    #             else:
    #                 filled_question = qas["question"].replace("多少", qas["answers"][0]["text"])
    #                 test_a = context_lst[i] + re.findall(p_before,filled_question)[0]
    #                 test_b = re.findall(p_after, filled_question)[0]
    #                 ans = re.findall(p_ans, qas["answers"][0]["text"])  

    #                 example = InputExample(
    #                     guid=str(guid),
    #                     text_a=test_a,
    #                     text_b=test_b,
    #                     tgt_text=ans)
                    
    #                 examples.append(example)
    #                 guid += 1
    #                 case_is_deleted = False
            
    #         if case_is_deleted:
    #             deleted_case += 1

    #     print("Finished:\n\tGot {} {} examples.\n Delete {} unanswerable questions, {} complicated questions.".format(str(guid), split, deleted_impossible_qas, deleted_complicated_qas + deleted_not_num_qas))

    #     return examples

PROCESSORS = {
    "cjrc_2019": CJRCProcessor,
    # "cjrc_2019_mask": CJRCMLMProcessor,
}
