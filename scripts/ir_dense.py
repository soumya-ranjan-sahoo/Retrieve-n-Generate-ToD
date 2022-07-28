from sentence_transformers import SentenceTransformer
import json
from utils.faiss_util import DenseHNSWFlatIndexer
from datetime import datetime


class IRindex:
    def __init__(self, dim = 768):
        self.encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.indexer = None
        self.i2record = None
        self.dim = dim

    def set_indexer(self, idxpath, fname=None):
        self.indexer = DenseHNSWFlatIndexer(1)
        self.indexer.deserialize_from(idxpath)
        self.i2record = json.load(open(idxpath[:idxpath.rfind("/")+1]+fname+"_i2record.json"))

    def lookup(self, text, topk=1):
        """
        Perform faiss_hnsw lookup
        :param topk: number of candidate entities
        :param text: text chunk to be looked up
        :return:  [labels],[ids] --> list of entity labels and entity ids
        """
        query_vector = self.encoder.encode([text])
        print(topk)
        sc, e_id = self.indexer.search_knn(query_vector, topk)  # 1 -> means top entity
        print("sc, e_id-",sc, e_id)
        try:
            ids, scores = [self.i2record[str(e_id[0][i])] for i in range(topk)], [sc[0][i] for i in range(topk)]
            return {
                "entries": ids,
                "scores": scores
            }
        except:
            return {
                "entries": [],
                "scores": []
            }


    def index_content(self, data, fname=None):
        encoded_data = self.encoder.encode(data)
        index = DenseHNSWFlatIndexer(self.dim, len(data))
        index.index_data(encoded_data)
        index.serialize(fname)

        id2entry = {str(i):val for i,val in enumerate(data)}
        json.dump(id2entry, open(fname[:fname.rfind("/")+1]+"i2record.json", "w"), indent=3)