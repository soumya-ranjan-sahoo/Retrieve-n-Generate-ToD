import numpy as np
import json
from nltk import bigrams as get_bigrams
from nltk import trigrams as get_trigrams
from nltk import word_tokenize, ngrams
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from collections import Counter
import math

from utils.data import normalize


def get_fourgrams(sequence, **kwargs):
    """
    Return the 4-grams generated from a sequence of items, as an iterator.

    :param sequence: the source data to be converted into 4-grams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 4, **kwargs):
        yield item


class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def update(self, output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


class UnigramMetric(Metric):
    def __init__(self, metric):
        self._score = None
        self._count = None
        if metric.lower() not in ["recall", "precision"]:
            raise ValueError(
                "mertic should be either 'recall' or 'precision', got %s" % metric)
        self.metric = metric.lower()
        super(UnigramMetric, self).__init__()

    def reset(self):
        self._score = 0
        self._count = 0
        super(UnigramMetric, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        hyp_tokens = normalize(hypothesis).split()
        ref_tokens = normalize(reference).split()

        common = Counter(ref_tokens) & Counter(hyp_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            score = 0
        else:
            if self.metric == "precision":
                score = 1.0 * num_same / len(hyp_tokens)
            else:
                assert self.metric == "recall"
                score = 1.0 * num_same / len(ref_tokens)

        self._score += score
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "Unigram metrics must have at least one example before it can be computed!")
        return self._score / self._count

    def name(self):
        return "Unigram{:s}".format(self.metric.capitalize())


class NGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n
        self._diversity = None
        self._count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("NGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")

        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(NGramDiversity, self).__init__()

    def reset(self):
        self._diversity = 0
        self._count = 0
        super(NGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output

        if hypothesis is None:
            diversity = 0
        else:
            diversity = 0
            output_tokens = word_tokenize(hypothesis)
            denominator = float(len(output_tokens))

            if denominator != 0.0:
                ngrams = set(list(self.ngram_func(output_tokens)))
                diversity = len(ngrams) / denominator

        self._diversity += diversity
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "NGramDiversity must consume at least one example before it can be computed!")
        return self._diversity / self._count

    def name(self):
        return "{:d}GramDiversity".format(self._n)


class CorpusNGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n

        self._ngrams = None
        self._token_count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("CorpusNGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")
        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(CorpusNGramDiversity, self).__init__()

    def reset(self):
        self._ngrams = set()
        self._token_count = 0
        super(CorpusNGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output
        if isinstance(hypothesis, str) and hypothesis:
            output_tokens = word_tokenize(hypothesis)

            ngrams = list(self.ngram_func(output_tokens))
            self._ngrams.update(ngrams)
            self._token_count += len(output_tokens)

    def compute(self):
        if self._token_count == 0:
            raise ValueError(
                "CorpusNGramDiversity must consume at least one example before it can be computed!")

        return len(self._ngrams) / self._token_count

    def name(self):
        return "Corpus{:d}GramDiversity".format(self._n)


class BLEU(Metric):
    def __init__(self, dataset):
        self.dataset = dataset
        self._bleu = None
        self._count = None
        self.domain = {
            "camrest": {

            },
            "incar": {
                "schedule": {
                    "scores": list(),
                    "count": 0
                },
                "navigate": {
                    "scores": list(),
                    "count": 0
                },
                "weather": {
                    "scores": list(),
                    "count": 0
                }
            },
            "woz2_1": {
                "attraction": {
                    "scores": list(),
                    "count": 0
                },
                "restaurant": {
                    "scores": list(),
                    "count": 0
                },
                "hotel": {
                    "scores": list(),
                    "count": 0
                }
            }
        }
        super(BLEU, self).__init__()

    def reset(self):
        self._bleu = 0
        self._count = 0
        super(BLEU, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference, task = output

        hyp_tokens = hypothesis.split()
        ref_tokens = reference.split()

        bleu = sentence_bleu([ref_tokens], hyp_tokens)
        if self.dataset == "incar" or self.dataset == "woz2_1":
            self.domain[self.dataset][task]["scores"].append(bleu)
            self.domain[self.dataset][task]["count"] += 1
        self._bleu += bleu
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "BLEU-1 must have at least one example before it can be computed!")

        if self.dataset == "incar" or self.dataset == "woz2_1":
            for k, v in self.domain[self.dataset].items():
                print(self.dataset, k, sum(v["scores"])/v["count"])
        return self._bleu / self._count

    def name(self):
        return "BLEU"


class BLEU2(Metric):
    def __init__(self):
        self._bleu = None
        self._count = None
        self.count = [0, 0, 0, 0]
        self.clip_count = [0, 0, 0, 0]
        self.r = 0
        self.c = 0
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.refs = list()
        self.hyps = list()
        super(BLEU2, self).__init__()

    def reset(self):
        self._bleu = 0
        self._count = 0
        super(BLEU2, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output
        self.hyps.append(hypothesis)
        self.refs.append(reference)

    def compute(self):
        for idx, refs in enumerate(self.refs):
            hyps = [hyp.split() for hyp in self.hyps[idx]]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    self.count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(
                                max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict(
                        (ng, min(count, max_counts[ng]))
                        for ng, count in hypcnts.items()
                    )
                    self.clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                self.r += bestmatch[1]
                self.c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        if self.c > 0:
            bp = 1 if self.c > self.r else math.exp(
                1 - float(self.r) / float(self.c))
        else:
            print("bp is 0", flush=True)
            bp = 1
        p_ns = [float(self.clip_count[i]) / float(self.count[i] + p0) +
                p0 for i in range(4)]
        s = math.fsum(w * math.log(p_n)
                      for w, p_n in zip(self.weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu

    def name(self):
        return "BLEU2"


class METEOR(Metric):
    def __init__(self):
        self._meteor = None
        self._count = None
        super(METEOR, self).__init__()

    def reset(self):
        self._meteor = 0
        self._count = 0
        super(METEOR, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        meteor = single_meteor_score(
            reference, hypothesis, preprocess=normalize)

        self._meteor += meteor
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "METEOR must have at least one example before it can be computed!")
        return self._meteor / self._count

    def name(self):
        return "METEOR"


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS

    This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)]
               for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class EntityF1:
    def __init__(self, dataset):
        self.dataset = dataset
        self.score = 0.0
        self.count = 0
        self.domain = {
            "camrest": {

            },
            "incar": {
                "schedule": {
                    "scores": list(),
                    "count": 0
                },
                "navigate": {
                    "scores": list(),
                    "count": 0
                },
                "weather": {
                    "scores": list(),
                    "count": 0
                }
            },
            "woz2_1": {
                "attraction": {
                    "scores": list(),
                    "count": 0
                },
                "restaurant": {
                    "scores": list(),
                    "count": 0
                },
                "hotel": {
                    "scores": list(),
                    "count": 0
                }
            }
        }
        self.entities = self.get_global_entities(dataset=dataset)

    def get_global_entities(self, dataset="incar"):
        if dataset == "incar":
            with open('data/incar/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_')
                                               for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_')
                                                   for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
            return global_entity_list
        elif dataset == "camrest" or dataset == "woz2_1":
            return json.load(open(f"data/{dataset}/entities.json"))

    def compute_prf(self, gold, pred, kb_plain):
        #local_kb_word = [k[2] for k in kb_plain]
        local_kb_word = kb_plain  # list(set(local_kb_word))
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in self.entities or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / \
                float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def update(self, output):
        pred, ref, kb, task = output
        kb_temp = kb
        f1, c = self.compute_prf(gold=ref, pred=pred, kb_plain=kb_temp)

        if self.dataset == "incar" or self.dataset == "woz2_1":
            self.domain[self.dataset][task]["scores"].append(f1)
            self.domain[self.dataset][task]["count"] += 1

        self.score += f1
        self.count += c

    def compute(self):
        if self.dataset == "incar" or self.dataset == "woz2_1":
            for k, v in self.domain[self.dataset].items():
                print(self.dataset, k, sum(v["scores"])/v["count"])
        return self.score/(self.count+1e-30)

    def name(self):
        return "Entity-F1"


class Rouge:
    """
    Class for computing ROUGE-L score for a set of candidate sentences

    This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    with minor modifications
    """

    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate.split()

        for reference in refs:
            # split into tokens
            token_r = reference.split()
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            try:
                prec.append(lcs / float(len(token_c)))
                rec.append(lcs / float(len(token_r)))
            except:
                prec =[0]
                rec = [0]

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / \
                float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def method(self):
        return "Rouge"


class ROUGE(Metric):
    def __init__(self):
        self.scorer = Rouge()
        self._rouge = None
        self._count = None
        super(ROUGE, self).__init__()

    def reset(self):
        self._rouge = 0
        self._count = 0
        super(ROUGE, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        rouge = self.scorer.calc_score(hypothesis, [reference])

        self._rouge += rouge
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError(
                "ROUGE-L must have at least one example before it can be computed!")
        return self._rouge / self._count

    def name(self):
        return "ROUGE"
