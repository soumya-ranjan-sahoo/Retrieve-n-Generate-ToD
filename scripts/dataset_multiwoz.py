import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import pad_ids, truncate_sequences
from itertools import chain
from tqdm import tqdm
import os
import numpy as np
from os.path import join
import json
import pickle
from scripts.ir_sparse import *
from nltk.corpus import stopwords
SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "[PAD]",
    "additional_special_tokens": ["[SYS]", "[USR]", "[KG]", "[SUB]", "[PRED]", "[OBJ]", "[TRIPLE]", "[SEP]", "[Q]","[DOM]"]
}

SPECIAL_TOKENS_VALUES = ["[BOS]", "[EOS]", "[PAD]", "[USR]", "[SUB]",
                         "[SYS]", "[USR]", "[SUB]", "[PRED]", "[OBJ]", "[TRIPLE]", "[SEP]", "[Q]","[DOM]"]


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        self.args = args

        # initialize indexing method
        if self.args.ir == "dense":
            print("*** Using Dense Retrieval ***")
            from scripts.ir_dense import IRindex
            self.indexer = IRindex()

        if self.args.ir == "sparse":
            print("*** Using Sparse Retrieval ***")

        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["pad_token"])
        self.sys_token, self.usr_token, self.kg, self.sub_token, self.pred_token, self.obj_token, self.triple_token, self.sep, self.ques,self.domain = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"])
        self.dialogs = self._prepare_conversations(
            dataset=name, split_type=split_type)
        self._create_examples()

    def build_input_from_segments(self, knowledge, history, response, example, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        sequence = [[self.bos] + knowledge] + history + \
            [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [[self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token] + s
                                 for i, s in enumerate(sequence[1:])]  # From the history

        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s)
                                              for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["pos_ids"] = [j for j in range(len(instance["input_ids"]))]
        return instance, sequence

    def _prepare_conversations(self, dataset="woz2_1", split_type="train"):
        print("Loading dialogue data...")
        formatted_dialogs = json.load(
            open(join("data", dataset, split_type+".json")))
        return formatted_dialogs

    def kg2seq(self, kg):
        """
        kg: a knowledge sequnece.  i.e, r1, o1, r2 o2 ....
        returns a knowledge sequence including special tokens.  i.e, [PRED] r1 [SUB] o1 [PRED] r2 [PRED] o2
        """
        return " ".join(["[PRED] "+f if i % 2 == 0 else "[SUB] "+f for i, f in enumerate(kg.split())]).strip()

    def match_kgentry(self, kg, gold):
        """
        kg: all the knowledge in the form of dictionary as it is for instance the kg in val.json
        gold: reference entity list
        """
        count = list()
        for i, entry in enumerate(kg):
            current_count = 0
            for ent in gold:
                if ent in list(entry.values()):
                    current_count += 1
            count.append(current_count)
        relevant_entry = kg[np.argmax(np.array(count))]
        kgitem = " "
        for k, v in relevant_entry.items():
            kgitem += " "+k+" "+v
        # returns the top most relevant knowledge in the from of sequece. i.e., [PRED] r1 [SUB] o1 [PRED] r2 [PRED] o2
        return self.tokenizer.tokenize(kgitem)

    def _create_examples(self):
        print("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs):
            if dialog["kg"]:
                if self.split_type == "train":    # check for the gold knowledge entry using the information from reference entities
                    # print("------------------------------------------")
                    used_knowledge = self.match_kgentry(dialog["kg"], dialog["ref_ents"])
                    
                else:
                    # during the evalution using IR methods to fetch the relevant topk knowledge
                    if self.args.ir == "dense":
                        idx_path = join("data/woz2_1", self.split_type,
                                        str(dialog["id"])+"_"+dialog["task"]+".pkl")
                        self.indexer.set_indexer(idxpath=idx_path, fname=str(
                            dialog["id"])+"_"+dialog["task"])
                        # last item of the history is the current user utterance
                        used_knowledge = self.indexer.lookup(
                            dialog["history"][-1], topk=self.args.topk)["entries"]
                        used_knowledge = ''.join(used_knowledge)
                        used_knowledge = " "+used_knowledge
                        #used_knowledge = [self.tokenizer.tokenize(entry)for entry in used_knowledge] # overwritten
                        used_knowledge = self.tokenizer.tokenize(used_knowledge)
                        

                    if self.args.ir == "sparse":

                        numberRelevantDocs = self.args.topk
                        print(numberRelevantDocs)
                        relevanceDict = {}
                        collectionList = CreateNewCollection(dialog)
                        queryHistory = CreateNewHistory(dialog)
                        queryHistory = [queryHistory[-1]] if len(queryHistory) == 1 else queryHistory[-2:]  # last 2 utterance
                        if collectionList == 0 or queryHistory == 0:
                            relevanceDict[n] = []  # KB/Utterance Not Available
                        else:
                            DF = CreateDF(collectionList)
                            tf_idf = CalculateTFIDF(DF, collectionList)
                            relevantDocDict = {}
                            docMatrix = tf_idf
                            for idx, query in enumerate(queryHistory):
                                query = query.lower().split()
                                query = [
                                    word for word in query if word not in stopwords.words('english')]
                                relevantDocDict[str(query)] = RelevantDocs(
                                    query, docMatrix)
                            docID = KRelevantRecords(
                                relevantDocDict, numberRelevantDocs)
                            relevanceDict[dialog["id"]] = docID
                        Dict = relevanceDict
                        for key, vals in Dict.items():
                            kgitem = " "
                            for val in vals:
                                knowledge = dialog['kg'][val]
                                for k, v in knowledge.items():
                                    kgitem += " "+k+" "+v
                        #used_knowledge = [self.kg2seq(kgitem.strip())] #over-written
                        #used_knowledge = used_knowledge[0].split() #over-written
                        used_knowledge = self.tokenizer.tokenize(kgitem)
                
               # used_knowledge = ["[DOM] " + dialog["task"]]+used_knowledge
                print("tokens",used_knowledge)
                used_knowledge = self.tokenizer.convert_tokens_to_ids(used_knowledge)
                print("ids",used_knowledge)
                # break
            else:
                used_knowledge = []

            dialog_id = dialog["id"]
            ref_ents = dialog["ref_ents"]

            # used_knowledge= [self.kg2seq(record) for record in dialog["kg"]]  # integrate IR here
            #used_knowledge = [self.ground_record(ref_ents, dialog["kg"])]
            used_knowledge_text = dialog["kg"]

            used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            history = [self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(turn)) for turn in dialog["history"]]
            gt_resp = dialog["response"]
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left
            truncated_history = truncate_sequences(
                truncated_history, self.args.history_max_tokens)

            self.examples.append({
                "history": truncated_history,
                "task": dialog["task"],
                "knowledge": used_knowledge,
                "knowledge_text": used_knowledge_text,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "dialog_id": dialog_id,
                "reference_entities": ref_ents
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class Dataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(Dataset, self).__init__(args, tokenizer,
                                      name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"], example["history"], example["response"], example)
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        pos_ids = [ins["pos_ids"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        pos_ids = torch.tensor(pad_ids(pos_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        return input_ids, pos_ids, lm_labels


class EvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(EvalDataset, self).__init__(
            args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
