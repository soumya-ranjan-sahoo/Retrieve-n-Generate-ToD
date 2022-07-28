import json
import os
import ast
from os.path import join
from tqdm import tqdm
import sys

def process_incar():
    print("In preprocess_conversation python file")
    splits = ["val","test","train"]
    dataroot = "data/incar"
    for sp in splits:

        with open(join(dataroot,"kvr",sp+".txt")) as f:
            conv_id = 0
            data = dict()
            task = None
            for line in tqdm(f, desc=f"{sp}:"):
                line = line.strip()
                if line:
                    if '#' in line:
                        line = line.replace("#", "")
                        task = line
                        conv_id += 1
                        data[conv_id] = {
                            "task": task,
                            "utterances": [],
                            "kg": []
                        }
                        continue

                    nid, line = line.split(' ', 1)

                    if '\t' in line:        # conversation
                        u, r, gold_ent = line.split('\t')
                        gold_ent = ast.literal_eval(gold_ent)
                        data[conv_id]["utterances"].append({
                            "user": u,
                            "response": r,
                            "reference_entities": gold_ent
                        })
                    else:                   # kg triples
                        triple = line.split()
                        if task=="weather":
                            if len(triple)==4:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[2]+" "+triple[3]])
                            elif len(triple)==2:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[0]])
                            else:
                                data[conv_id]["kg"].append(triple)
                        else:
                            if len(triple)==3:
                                data[conv_id]["kg"].append(triple)

            json.dump(data, open(join(dataroot,sp+".json"), "w"), indent=3)


def process_data(dataset="incar"):
    if dataset=="incar":
        process_incar()

if __name__=="__main__":
    process_data(dataset="incar")