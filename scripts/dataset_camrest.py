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
import json
from flair.data import Sentence
from flair.models import SequenceTagger
import re

# load tagger
tagger = SequenceTagger.load("flair/upos-english-fast")

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
        self.postag_mapping = json.load(open("postag_mapping.json"))
        self.postag_dict = {v:int(k) for k,v in self.postag_mapping.items()}
        self._create_examples()

    def build_input_from_segments(self, knowledge, history, response, tag_knowledge, tag_history, tag_response, example, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        #print("values:",response, tag_response)
        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        #print("sequence - ",sequence)
        sequence_with_speaker = [[self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token] + s
                                 for i, s in enumerate(sequence[1:])]  # From the history
        #print("sequence_with_speaker - ",sequence)
        sequence = [sequence[0]] + sequence_with_speaker
        #print("sequence len",len(sequence))
        instance["input_ids"] = list(chain(*sequence))

       
        ### postag sequence ###
        tag_sequence = [[self.bos] + tag_knowledge] + tag_history + \
            [tag_response + ([self.eos] if with_eos else [])]
        #print("tag_sequence - ",tag_sequence)
        tag_sequence_with_speaker = [[self.usr_token if ((len(tag_sequence)-i) % 2) == 0 else self.sys_token] + s
                                 for i, s in enumerate(tag_sequence[1:])]  # From the history
        #print("tag_sequence_with_speaker - ",tag_sequence_with_speaker)
        tag_sequence = [tag_sequence[0]] + tag_sequence_with_speaker
        #print("tag_sequence len",len(tag_sequence))
        instance["postag_ids"] = list(chain(*tag_sequence))



        
        #print("decoding input_ids",self.tokenizer.decode( instance["input_ids"]))
        #print("instance_input_ids",instance)
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s)
                                              for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        #print("instance_lm_labels",instance)
        instance["pos_ids"] = [j for j in range(len(instance["input_ids"]))]

        try:
            assert len(instance["input_ids"])==len(instance["postag_ids"])==len(instance["pos_ids"])
        except:
            print("ALT: ",len(instance["input_ids"]),len(instance["postag_ids"]),len(instance["pos_ids"]))


        return instance, sequence

    def _prepare_conversations(self, dataset="camrest", split_type="train"):
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
       # return self.kg2seq(kgitem.strip()) #over-written
        return self.tokenizer.tokenize(kgitem)
 
    
    def postag_knowledge(self,tokenizedKnowledge):
        ''' Creates Postags for knowledge'''
        knowledge = [token.replace('Ġ', '') for token in tokenizedKnowledge] ### Resomving the Ġ symbol from the list
        knowledge = Sentence(knowledge)
        tagger.predict(knowledge)
        postagKnowledgeList = re.findall("\<(.*?)\>", str(knowledge))
        if tokenizedKnowledge[0] == "Ġ":
       	  postagKnowledgeList.insert(0,'X')
        return postagKnowledgeList



    def postag_history(self,tokenizedHistory):
        ''' Creates Postags for History'''
        history = [token.replace('Ġ', '') for token in tokenizedHistory] ### Resomving the Ġ symbol from the list
        history = Sentence(history)
        tagger.predict(history)
        postagHistory = re.findall("\<(.*?)\>", str(history))
        return postagHistory


    def postag_response(self,response):
        ''' Creates Postags for response'''
        #response = [token.replace('Ġ', '') for token in response] ### Removing the Ġ symbol from the list
        response = Sentence(response)
        tagger.predict(response)
        postagResponseList = re.findall("\<(.*?)\>", str(response))
        return postagResponseList
    

    def _create_examples(self):
        print("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs):
            try:
                if dialog["kg"]:
                    if self.split_type == "train":    # check for the gold knowledge entry using the information from reference entities
                        #print("checking testset with groundtruth")
                        used_knowledge = self.match_kgentry(dialog["kg"], dialog["ref_ents"])
                        # print(dialog["id"])
                        #print("Gold ent: ", dialog["ref_ents"])
                        #print("selected entry: ", used_knowledge)
                    else:
                        # during the evalution using IR methods to fetch the relevant topk knowledge
                        #print("checking evalsets") 
                        if self.args.ir == "dense":
                            idx_path = join("data/camrest", self.split_type,
                                            str(dialog["id"])+"_"+dialog["task"]+".pkl")
                            self.indexer.set_indexer(idxpath=idx_path, fname=str(
                                dialog["id"])+"_"+dialog["task"])
                            # last item of the history is the current user utterance
                            used_knowledge = self.indexer.lookup(
                                dialog["history"][-1], topk=self.args.topk)["entries"]
                            used_knowledge = ''.join(used_knowledge)
                            used_knowledge = " "+used_knowledge
                            #used_knowledge = [self.tokenizer.tokenize(entry)for entry in used_knowledge] # over-written
                            used_knowledge = self.tokenizer.tokenize(used_knowledge)
                            

                        if self.args.ir == "sparse":

                            numberRelevantDocs = self.args.topk
                            print(numberRelevantDocs)
                            relevanceDict = {}
                            collectionList = CreateNewCollection(dialog)
                            queryHistory = CreateNewHistory(dialog)
                            queryHistory = [queryHistory[-1]] if len(queryHistory) == 1 else queryHistory[-2:]  # last 2 utterance
                            if collectionList == 0 or queryHistory == 0:
                                relevanceDict[dialog["id"]] = []  # KB/Utterance Not Available
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
                    #print("tokens",used_knowledge)
                    used_knowledge_tokens = used_knowledge
                    #print("used_knowledge_type",type(used_knowledge))
                    #print("used_knowledge_length",len(used_knowledge))
                    used_knowledge = self.tokenizer.convert_tokens_to_ids(used_knowledge)
                    #print("ids",used_knowledge)
                    # break
                else:
                    used_knowledge = []
                    used_knowledge_tokens = used_knowledge
                    

                dialog_id = dialog["id"]
                ref_ents = dialog["ref_ents"]

                
                used_knowledge_text = dialog["kg"] # has not been tokenized when appended as examples
                #used_knowledge_tokens =[]
                #used_knowledge = []

                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
                history = [self.tokenizer.tokenize(turn) for turn in dialog["history"]]
                history_tokens_list = history
                #print("---------------History-----------------")
                #print(history,type(history),len(history))
                history = [self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(turn)) for turn in dialog["history"]]
                #print("history_ids",history)
                gt_resp = dialog["response"]
                tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(gt_resp))
                #print("tokenized_gt_resp",tokenized_gt_resp)
                #print("tokenized_gt_resp_decoded",self.tokenizer.decode(tokenized_gt_resp),len(self.tokenizer.decode(tokenized_gt_resp)))
                #print("tokenized_gt_resp_decoded w/o special tokens",self.tokenizer.decode(tokenized_gt_resp,skip_special_tokens=True),len(self.tokenizer.decode(tokenized_gt_resp,skip_special_tokens=True)))

                # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
                truncated_history = history[-self.args.history_max_utterances:]
                #print("truncated_history_ids",truncated_history)
                # perform token-level truncation of history from the left
                truncated_history = truncate_sequences(
                    truncated_history, self.args.history_max_tokens)

                ### Post-tagging ###
            
                if used_knowledge_tokens:
                    postagKnowledge = self.postag_knowledge(used_knowledge_tokens)
                   # print("catch error----", len(postagKnowledge), len(used_knowledge_tokens), postagKnowledge, used_knowledge_tokens)
                    assert len(postagKnowledge) == len(used_knowledge_tokens), "Mismatch of knowledge ids with knowledge tokens"
                    postagKnowledgeIds = [self.postag_dict.get(item,item)  for item in postagKnowledge]
                    postagKnowledgeIds = postagKnowledgeIds[:self.args.knowledge_max_tokens] # truncation
                else:
                    postagKnowledgeIds = []

                assert len(postagKnowledgeIds) == len(used_knowledge),"Mismatch of knowledge ids with knowledge pos ids tokens"

                postagHistoryList = []
                if history_tokens_list:
                    for tokenizedHistory in history_tokens_list:
                        #print("postagHistoryList",len(tokenizedHistory))
                        tokens = self.postag_history(tokenizedHistory)
                        if len(tokens) - len(tokenizedHistory) == 1:
                            del tokens[0]
                        postagHistoryList.append(tokens)
                        assert len(tokenizedHistory) == len(tokens),"Mismatch of history ids with history tokens"
                        #print("postag_history",len(postag_history(tokenizedHistory)))
                # print(postagHistoryList)
                    postagHistoryIds = [[self.postag_dict.get(item,item) for item  in postagHistory] for postagHistory in postagHistoryList]
                # print(postagHistoryIds)
                    postagHistoryIds = postagHistoryIds[-self.args.history_max_utterances:]# truncation
                # print(postagHistoryIds,len(postagHistoryIds),truncated_history,len(truncated_history))
                    if len(postagHistoryIds) != len(truncated_history): print("mismatch lengths:",len(postagHistoryIds),"and",len(truncated_history))
                    assert len(postagHistoryIds) == len(truncated_history),"Mismatch of history ids with history pos ids tokens"
                else:
                    postagHistoryIds = []

                
                ### Error in the next section of the code ###
                ##### len(tokenized_gt_resp) has more tokens than it actually should have i.e. we need to understand what are the additional tokens for 
                # and also see if we are missing out to add some more tokens while doing postaging which doesn't seem to be the case when checking manually --> Need to Indetify #####

                if tokenized_gt_resp:
                    postagResponse = self.postag_response(self.tokenizer.tokenize(dialog["response"]))
                    #postagResponse = self.postag_response(self.tokenizer.decode(tokenized_gt_resp,skip_special_tokens=True))
                    try:
                        assert len(postagResponse) == len(tokenized_gt_resp), "Mismatch of Response ids with Response pos ids tokens"   #Error: tokenized_gt_resp has more additional tokens that we need to indentify 
                    except:
                        print("Lengths:", len(postagResponse),len(tokenized_gt_resp))
                    postagResponseIds = [self.postag_dict.get(item,item)  for item in postagResponse]
                else:
                    postagResponseIds = []
            



                self.examples.append({
                    "history": truncated_history,
                    "task": dialog["task"],
                    "knowledge": used_knowledge,
                    "knowledge_text": used_knowledge_text,
                    "response": tokenized_gt_resp,
                    "response_text": gt_resp,
                    "dialog_id": dialog_id,
                    "reference_entities": ref_ents,
                    "postag_knowledge": postagKnowledgeIds,
                    "postag_history": postagHistoryIds,
                    "postag_response": postagResponseIds
                })
            except Exception as e:
                print(e)
                pass


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
            example["knowledge"], example["history"], example["response"], example["postag_knowledge"], example["postag_history"], example["postag_response"], example)
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        pos_ids = [ins["pos_ids"] for ins in batch]
        postag_ids = [ins["postag_ids"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        pos_ids = torch.tensor(pad_ids(pos_ids, self.pad))
        postag_ids = torch.tensor(pad_ids(postag_ids, 0))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        print("Sizes",postag_ids.size(), input_ids.size(), lm_labels.size(),pos_ids.size())
        return input_ids, pos_ids,postag_ids, lm_labels


class EvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(EvalDataset, self).__init__(args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
