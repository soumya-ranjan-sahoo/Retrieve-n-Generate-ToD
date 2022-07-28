#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os.path import join
import json
from tqdm import tqdm
import re
from transformers import pipeline
import os
from os.path import join
import json
from tqdm import tqdm
import re
os.chdir("path-to-your-directory") ### path of your directory 
import time
import json
import re
import os 
import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
#from num2words import num2words
import string
import numpy as np
import copy
import pandas as pd
from nltk.corpus import stopwords

def convert_lower_case(data):
    return data.lower()

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    data = re.sub(r'["!\"#$%&()*+-./:;<=>?@[\]^`{|}~\n"]',' ',data)
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    #data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) 
    return data




def CreateCollection(KBNumber,json_data):
        """
        Creates collection for the IR Task
        Collection here is the set of KBs associated with a specific conversation
        Each document in the collection is representative of indivual KB. 
        Parameters :
        KBNumber - Number of the KG ; Used for indexing a specific KB
        json_data - The input file that has the KBs in it
    
        """ 
        try:
          
            KB = json_data[KBNumber]["kg"]
            collectionList =[]
            for record in KB:
                dictList=[]
                for key, value in record.items():
                    #print("key-",key,"Value-",value)
                    dictList.append(key+":"+value)
                collectionList.append(dictList)
            return collectionList  
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            return 0
        

def CreateNewCollection(dialog):
        """
        %Currently in use --- V2.0%
        Creates collection for the IR Task
        Collection here is the set of KBs associated with a specific conversation
        Each document in the collection is representative of indivual KB. 
        Parameters :
        KBNumber - Number of the KG ; Used for indexing a specific KB
        json_data - The input file that has the KBs in it
    
        """ 
        try:
          
            KB = dialog["kg"]
            collectionList =[]
            for record in KB:
                dictList=[]
                for key, value in record.items():
                    #print("key-",key,"Value-",value)
                    dictList.append(key+":"+value)
                collectionList.append(dictList)
            return collectionList  
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            return 0
        
        
def CreateConversation(convNumber,json_data):
        """
        Conversation history and gold utterance are created here.
        Note : We use the current utterance too as history
        Parameters :
        convNumber - Number of the Conversation ; should be same as KBNumber in CreateCollection()
        json_data - The input file that has the KBs in it
    
        """ 
    
        queryConversation = []
        goldUtterance = []
        try:
            for conversation in json_data[convNumber]["dialogue"]:
                
                if (conversation["data"]["end_dialogue"]) != True:
                    queryConversation.append((conversation["data"]["utterance"]).lower())
                else:
                    goldUtterance.append((conversation["data"]["utterance"]).lower())
                    break
            return queryConversation, goldUtterance
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            return 0, 0
            
def SlidingWindow(convNumber, json_data, windowSize):
        """
        Creates a Sliding Window of size windowSize for the IR Task which would be a query
        Moving window is created using the conversation history and the current utterance
        Parameters :
        convNumber - Number of the Conversation ; should be same as KBNumber in CreateCollection()
        json_data - The input file that has the KBs in it
        windowSize - Moving Window size for the utterance 
    
        """ 
        queryConversation, goldUtterance = CreateConversation(convNumber,json_data)
        queries = []
        for conversation in queryConversation:
            #print(conversation)
            #conversation = preprocess(conversation)
            split_sequence = conversation.lower().split()
            iteration_length = len(split_sequence) - (windowSize - 1)
            max_window_indicies = range(iteration_length)
            for index in max_window_indicies:
                queries.append(split_sequence[index:index + windowSize])
        return queryConversation, goldUtterance, queries
        
        
    


### DOCUMENT FREQUENCY UTILITY ###

def CreateDF(collectionList):
    DF = {}
    for i,kb in enumerate(collectionList):
        for record in kb:
            record = remove_punctuation(record)
            tokens = word_tokenize(remove_punctuation(record))
            for w in tokens:
                try:
                    DF[w].add(i)
                except:
                    DF[w] = {i}
    for i in DF:
        DF[i] = len(DF[i])
    
    return DF

    
def DocFreq(DF,word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c




### TF-IDF CALCULATOR ###

def CalculateTFIDF(DF,collectionList):
    
    
    docNo = 0
    N = len(collectionList)
    tf_idf = {}
    
    for i,kb in enumerate(collectionList):
    
        for record in kb:
            record = remove_punctuation(record)
            tokens = word_tokenize(remove_punctuation(record))
            counter = Counter(tokens)
            words_count = len(tokens)
            for token in np.unique(tokens):
                tf = counter[token]/words_count
                df = DocFreq(DF,token)
                idf = np.log((N+1)/(df+1))
                tf_idf[docNo, token] = tf
        docNo += 1
    assert (N == docNo)
    return tf_idf




### IDF CALCULATOR ###
def CalculateIDF(DF,collectionList):
    docNo = 0
    N = len(collectionList)
    _idf = {}
    for i,kb in enumerate(collectionList):
    
        for record in kb:
            record = remove_punctuation(record)
            tokens = word_tokenize(remove_punctuation(record))
            counter = Counter(tokens)
            words_count = len(tokens)
            for token in np.unique(tokens):
                #tf = counter[token]/words_count
                df = DocFreq(DF,token)
                idf = np.log((N+1)/(df+1))
                _idf[docNo, token] = idf
        docNo += 1
    assert (N == docNo)
    return _idf
    



### AVG-IDF CALCULATOR ###
def CalculateAvgIDF(collectionList,query):
    queryScore = 0
    N = len(collectionList)
    for token in query:
        df = DocFreq(DF,token)
        queryScore+= np.log((N+1)/(df+1))    
    return queryScore/3
    

    
def RelevantDocs(query,docMatrix):
    
    tokens = query
    query_weights = {}
    for key in docMatrix:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += docMatrix[key]
            except:
                query_weights[key[0]] = docMatrix[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)  
    docIds = []
    for i in query_weights[:10]:  ### selecting top 10 relevant docs
        docIds.append(i[0])
    return docIds
    

def KRelevantRecords(relevantDocDict,numberRelevantDocs):
    
    from collections import Counter
    
    flat_list = []
    docids = []
    try:
        flat_list = [item for sublist in list(relevantDocDict.values()) for item in sublist]
        data = Counter(flat_list)
        for docs in data.most_common(numberRelevantDocs):
            docids.append(docs[0])
        return docids    
        
    except:
        return []

def CreateHistory(n,json_data):
    return json_data[n]["history"]


def CreateNewHistory(dialog):
    """
    %Currently in use --- V2.0%
    """
    return dialog["history"]


def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    try:
        result = len(act_set & pred_set) / float(len(act_set))
    except:
        result = 0.0
    return result


def AverageScore(lst):
    return sum(lst) / len(lst)

def AverageF1(precision,recall):
    return 2*((precision*recall)/(precision+recall))

def DRIVERUTILITY(json_data,numberRelevantDocs):
    """
    Driver Method
    """
    N = len(json_data)
    print("The json files has %3d KBs"%(len(json_data)))
    relevanceDict = {}
    for n in range(6):
        collectionList = CreateCollection(n,json_data)
        #queryConversation, goldUtterance = CreateConversation(n,json_data)
        queryHistory = CreateHistory(n,json_data)
        if collectionList == 0 or queryHistory == 0:
            relevanceDict[n] = [] # KB/Utterance Not Available
        else:
            DF = CreateDF(collectionList)
            tf_idf = CalculateTFIDF(DF,collectionList)
            relevantDocDict = {}
            docMatrix = tf_idf
            for idx, query in enumerate(queryHistory):
                query = query.lower().split()
                relevantDocDict[str(query)] = RelevantDocs(numberRelevantDocs,query,docMatrix)
            docID = KRelevantRecords(relevantDocDict,numberRelevantDocs)
            relevanceDict[n] = docID
    return relevanceDict



def match_kgentry(kg, gold):
        """
        kg: all the knowledge in the form of dictionary as it is for instance the kg in val.json
        gold: reference entity list
        """
        count = list()
        print(kg)
        print(gold)
        for i, entry in enumerate(kg):
            current_count = 0
            for ent in gold:
                if ent in list(entry.values()):
                    current_count += 1
            count.append(current_count)
        return count

class BARTClassifier:
    def __init__(self, args_use_gpu):
        self.classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', device=0 if args_use_gpu else -1)

    def classify(self, sequence, candidate_labels, multi_class=True):
        print("In class method")
        return self.classifier(sequence, candidate_labels) if multi_class is None else self.classifier(sequence, candidate_labels, multi_class=True)

classifier = BARTClassifier(args_use_gpu = False)
    
    
def _prepare_conversations(dataset="woz2_1", split_type="train"):
        print("Loading dialogue data...")
        formatted_dialogs = json.load(open(join("data",dataset,split_type+".json")))
        return formatted_dialogs

dialogs = _prepare_conversations(dataset="woz2_1", split_type="train")
dialogs = dialogs
numberRelevantDocs = 2  ### RETRIVED DOCS
precision_at_k_list = []
recall_at_k_list = []
emptyListCount = 0
classifier = BARTClassifier(args_use_gpu = False)
for dialog in tqdm(dialogs):
        ref_entities = dialog["ref_ents"]
        actual = []
        predicted = []
        kg_list = []
        if dialog["kg"]:
            for i,kg in enumerate(dialog["kg"]):
                kg_list.append((' '.join(kg.values())))
                if all(ent in kg.values() for ent in ref_entities):
                    actual.append(i)    
            queryHistory = CreateNewHistory(dialog)
            #queryHistory = [queryHistory[-1]] if len(queryHistory) == 1 else queryHistory[-2:] ### QUERY SIZE - last 2 dialogues
            if queryHistory == 0:
                predicted = [] # KB/Utterance Not Available
            else:
                for idx, query in enumerate(queryHistory):
                    ans = classifier.classify(query,kg_list)
                    kg_relevant = ans["labels"][:numberRelevantDocs]
                #print(len(kg_relevant))
                for kg in kg_relevant:
                        predicted.append(kg_list.index(kg))
            if not actual or not predicted:
                emptyListCount+=1
            precision_at_k_list.append(precision_at_k(actual,predicted,numberRelevantDocs))
            recall_at_k_list.append(recall_at_k(actual,predicted,numberRelevantDocs))


print("------------------------------------------------------------------")
avg_precision = AverageScore(precision_at_k_list)
avg_recall = AverageScore(recall_at_k_list)
print("avg_precision",avg_precision) 
print("avg_recall",avg_recall) 
print("AverageF1",AverageF1(avg_precision,avg_recall))
print("emptyListCount",emptyListCount/len(dialogs))


# In[ ]:




