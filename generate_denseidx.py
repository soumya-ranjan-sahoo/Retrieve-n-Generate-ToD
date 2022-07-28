import json
from os.path import join
import os
from scripts.ir_dense import IRindex
from tqdm import tqdm
from utils.faiss_util import DenseHNSWFlatIndexer
import argparse
from argparse import ArgumentParser


def incar(indexer):
    splits = ["val","test","train"]
    dataroot = "data/incar"
    print("Indexing Incar")
    for sp in splits:
        os.makedirs(join(dataroot,sp), exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname) and d["kg"]:
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                encoded_data = indexer.encoder.encode(kg)
                index = DenseHNSWFlatIndexer(768, len(kg))
                index.index_data(encoded_data)
                index.serialize(pklname)

                id2entry = {str(i):val for i,val in enumerate(kg)}
                json.dump(id2entry, open(pklname[:pklname.rfind("/")+1]+fname+"_i2record.json", "w"), indent=3)
                

def camrest(indexer):
    splits = ["val","test","train"]
    dataroot = "data/camrest"
    print("Indexing Camrest")
    for sp in splits:
        os.makedirs(join(dataroot,sp), exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname) and d["kg"]:
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                encoded_data = indexer.encoder.encode(kg)
                index = DenseHNSWFlatIndexer(768, len(kg))
                index.index_data(encoded_data)
                index.serialize(pklname)

                id2entry = {str(i):val for i,val in enumerate(kg)}
                json.dump(id2entry, open(pklname[:pklname.rfind("/")+1]+fname+"_i2record.json", "w"), indent=3)

def multiwoz(indexer):
    splits = ["val","test","train"]
    dataroot = "data/woz2_1"
    print("Indexing Multiwoz")
    for sp in splits:
        os.makedirs(join(dataroot,sp), exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname) and d["kg"]:
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                encoded_data = indexer.encoder.encode(kg)
                index = DenseHNSWFlatIndexer(768, len(kg))
                index.index_data(encoded_data)
                index.serialize(pklname)

                id2entry = {str(i):val for i,val in enumerate(kg)}
                json.dump(id2entry, open(pklname[:pklname.rfind("/")+1]+fname+"_i2record.json", "w"), indent=3)
                #indexer.index_content(data=encoded_data, fname=pklname)

if __name__=="__main__":
    indexer = IRindex()
    parser = argparse.ArgumentParser()

    # Dataset parameter
    parser.add_argument("--dataset", type=str, default="woz2_1", choices=["incar","camrest", "woz2_1"],
                        help="dataset name.")
    args = parser.parse_args()

    print("Dataset processing:",  args.dataset)
    if args.dataset=="incar":
        incar(indexer)
    elif args.dataset=="camrest":
        camrest(indexer)
    elif args.dataset=="woz2_1":
        multiwoz(indexer)




    
