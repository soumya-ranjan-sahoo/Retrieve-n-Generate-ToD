import json
from os.path import join
import os
from scripts.ir_dense import IRindex


def incar(indexer):
    splits = ["val","test","train"]
    dataroot = "data/incar"
    for sp in splits:
        os.makedirs(path, exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname):
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                indexer.index_content(data=kg, fname=pklname)

def camrest(indexer):
    splits = ["val","test","train"]
    dataroot = "data/camrest"
    for sp in splits:
        os.makedirs(path, exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname):
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                indexer.index_content(data=kg, fname=pklname)

def multiwoz(indexer):
    splits = ["val","test","train"]
    dataroot = "data/woz2_1"
    for sp in splits:
        os.makedirs(path, exist_ok=True)
        data = json.load(open(join(dataroot,sp+".json")))
        for d in tqdm(data, desc=f"{sp}"):
            fname = str(d["id"])+"_"+d["task"]
            pklname = os.path.join(dataroot,sp,fname+".pkl")
            kg = list()
            if not os.path.isfile(pklname):
                for record in d["kg"]:
                    kgitem = " "
                    for k,v in record.items():
                        kgitem+=" "+k+" "+v
                    kg.append(kgitem.strip())
                indexer.index_content(data=kg, fname=pklname)

    
if __name__=="__main__":
    indexer = IRindex()

    if args.dataset=="incar":
        incar(indexer)
    elif args.dataset=="camrest":
        camrest(indexer)
    elif args.dataset=="woz2_1":
        multiwoz(indexer)