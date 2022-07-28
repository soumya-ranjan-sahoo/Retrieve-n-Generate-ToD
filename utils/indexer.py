from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import json
from faiss_util import DenseHNSWFlatIndexer
from datetime import datetime


def get_args():
    parser = ArgumentParser(description="Entity indexer")
    parser.add_argument("--output_path", required=True, type=str, help="output path")
    parser.add_argument("--faiss_index", type=str, default="hnsw", help='hnsw index')
    parser.add_argument('--index_buffer', type=int, default=50000)
    parser.add_argument("--save_index", action='store_true', help='save indexed file')
    parsed_args = parser.parse_args()
    #parsed_args = parsed_args.__dict__
    return parsed_args


def main(args):

    data, idx2info, idx2triple = list(), dict(), dict()

    start_time = datetime.now()
    if args.dataset == "incar":
        kg = json.load(open("data/incar/test.json"))["2"]["kg"]
    elif args.dataset == "camrest":
        kg = json.load(open("data/camrest/test.json"))["2"]["kg"]
    elif args.dataset == "woz2_1":
        kg = json.load(open("data/woz2_1/test.json"))["2"]["kg"]
        

    for i,entry in enumerate(kg):
        info = entry[0]+" "+entry[1]
        data.append(info)
        idx2info[i] = info
        idx2triple[i] = entry


    print('Loading Data {}'.format(datetime.now() - start_time))
    start_time = datetime.now()
    dataset =  args.dataset 
    print("In indexer and processing the dataset:", dataset)
    json.dump(idx2info,open(join("data",dataset,"processed_data"+"i2e.json"),"w"), ensure_ascii=False)
    json.dump(idx2triple,open(join("data",dataset,"processed_data"+"i2triple.json"),"w"), ensure_ascii=False)
    #json.dump(idx2triple, open("data/incar/processed_data/i2triple.json", "w"), ensure_ascii=False)

    start_time = datetime.now()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    print('Loaded Sentence Transformer: {}'.format(datetime.now() - start_time))

    start_time = datetime.now()
    encoded_data = model.encode(data)
    print('Completed encoding: {}'.format(datetime.now() - start_time))

    print("Using HNSW index in FAISS")
    vector_size = 768
    index = DenseHNSWFlatIndexer(vector_size, len(data))

    print("Building index.")
    start_time = datetime.now()
    index.index_data(encoded_data)
    print("Done indexing data.")
    print('Indexing Duration: {}'.format(datetime.now() - start_time))

    if args.save_index:                                                 # saving index
        print("Saving index file")
        index.serialize(args.output_path)
        print("Done")

if __name__ == '__main__':

    args = get_args()
    main(args)