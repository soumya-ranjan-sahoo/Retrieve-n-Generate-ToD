from transformers import pipeline
import os
from os.path import join
import json
from tqdm import tqdm
import re

os.chdir("C:\WorkDirectory\DialogueSystems\TaskOriented\Github\main\kg-structure-aware-dialogues")
def _prepare_conversations(dataset="woz2_1", split_type="train"):
        print("Loading dialogue data...")
        formatted_dialogs = json.load(open(join("data",dataset,split_type+".json")))
        return formatted_dialogs

dialogs = _prepare_conversations(dataset="woz2_1", split_type="train")
dialogs = dialogs[:10]
print("sguhskn")


class BARTClassifier:
    def __init__(self, args_use_gpu):
        self.classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', device=0 if args_use_gpu else -1)

    def classify(self, sequence, candidate_labels, multi_class=None):
        print("In class method")
        return self.classifier(sequence, candidate_labels) if multi_class is None else self.classifier(sequence, candidate_labels, multi_class=True)


classifier = BARTClassifier(args_use_gpu = False)
print("srfsygsh")
print(classifier.classify("there are 2 locations that match your criteria . the alpha-milton_guest_house with a 3_star rating and the avalon with a 4_star rating . do either of these sound good ?'",
'chiquito_restaurant_bar'))
print("abc")