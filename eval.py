import argparse
import glob
import logging
import os
import random
import shutil
import json
import nltk
nltk.download('wordnet')

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from scripts.model import GPT2LMHeadModel

from utils.args import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from scripts.model import run_batch_generation_sample
from utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, BLEU2, METEOR, ROUGE, EntityF1
)
from utils.data import write_generation_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def clean_text(text, args):
    #text =text.replace("?"," ?").replace(","," ,").replace("!"," !").strip()
    text = text.replace("."," .").replace("'"," ' ").replace("?"," ?").replace(","," ,").replace("!"," !").replace("  "," ").replace("you are","you ' re")
    if args.dataset=="incar":
        text = text.replace("p_ ._f_ ._changs","p_._f_._changs")
    #if text.endswith("."):
    #    text = text[:-1].strip()
    if args.dataset=="woz2_1":
        text = text.replace(".","")
    return text

def clean_gt(text):
    text = text.replace("you are","you ' re")
    return text

def evaluate(args, eval_dataset, model, tokenizer, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        print("Evaluating the dataset:",args.dataset)
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    metrics = [
        UnigramMetric(metric="recall"),
        UnigramMetric(metric="precision"),
        NGramDiversity(n=1),
        NGramDiversity(n=2),
        NGramDiversity(n=3),
        NGramDiversity(n=4),
        CorpusNGramDiversity(n=1),
        CorpusNGramDiversity(n=2),
        CorpusNGramDiversity(n=3),
        CorpusNGramDiversity(n=4),
        BLEU(dataset=args.dataset),
        METEOR(),
        ROUGE(),
        EntityF1(dataset=args.dataset)
    ]

    args.tokenizer = tokenizer
    all_output_texts = []
    all_ground_truths = []
    dialog_ids = []
    tasks = []
    decoded_knowledges = []
    do_evaluate = False
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            sampled_output_ids, ground_truth, dialog_id, ref_entities, knowledge, task = run_batch_generation_sample(args, model, batch, eval_dataset)
            sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
            tasks.append(task)
            sampled_output_text = clean_text(sampled_output_text,args)
            ground_truth = clean_gt(ground_truth)
            print("Dialog Id: ", dialog_id)
            print("Truth: ", ground_truth)
            print("Predt: ", sampled_output_text)
            decoded_knowledge =  tokenizer.decode(knowledge, skip_special_tokens=True)
            print("Knowledge used: ", decoded_knowledge)
            #ground_truth = clean_text(ground_truth,args)
            all_output_texts.append(sampled_output_text)
            all_ground_truths.append(ground_truth)
            dialog_ids.append(dialog_id)
            decoded_knowledges.append(decoded_knowledge)
        if ground_truth.strip() != "":
            do_evaluate = True
            for metric in metrics:
                if metric.name()=="Entity-F1":
                    if ref_entities:
                        metric.update((sampled_output_text, ref_entities, knowledge, task))
                elif metric.name()=="BLEU":
                    metric.update((sampled_output_text, ground_truth, task))
                else:
                    metric.update((sampled_output_text, ground_truth))

    if args.output_file:
        write_generation_preds(args.output_file, dialog_ids, all_output_texts, all_ground_truths,decoded_knowledges, tasks)

    result = dict()
    if do_evaluate and args.local_rank in [-1, 0]:                                         
        output_eval_file = os.path.join(eval_output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            #logger.info("***** Eval results %s *****" % desc)
            #writer.write("***** Eval results %s *****\n" % desc)
            logger.info("***** Eval results %s with K = %i and P = %2f *****" % (desc,args.top_k,args.top_p))
            writer.write("***** Eval results %s with K = %i and P = %2f *****\n" % (desc,args.top_k,args.top_p))
            for metric in metrics:
                name = metric.name()
                score = metric.compute()
                result[name] = score
                logger.info("  %s = %s", name, str(score))
                writer.write("%s = %s\n" % (name, str(score)))

    return result


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument("--decode", type=str, default="basic", choices=["basic","beam"], help="decoding technique")
    parser.add_argument('--dataset', type=str, default='incar', choices=['incar','camrest','woz2_1'])
    parser.add_argument("--generation_params_file", type=str, default="",
                        help="JSON configuration file for generation-related configurations.")
    parser.add_argument("--top_weights", type=int, default=-1,
                        help="Triples from top-k ent to put attention on (recommended 5")
    parser.add_argument("--dataroot", type=str, default="",
                        help="Path to dataset, will override the path in config.")
    parser.add_argument("--ir", type=str, default="sparse", choices=["dense","sparse"], help="retrieval method")
    parser.add_argument("--topk", type=int, default=3, help="topk KB records")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                             "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
                        
    parser.add_argument("--no_sample", type=bool, default=False,
                        help="related to sampling")
    parser.add_argument("--min_length", type=int, default=1,
                        help="Minimum length of the generated response")
    parser.add_argument("--max_length", type=int, default=80,
                        help="Maximum length of the generated response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Helps in logit's scaling by a factor of 1/n where n = value*100")
    parser.add_argument("--top_k", type=int, default=50,
                        help="K value for top k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="P value for nucleus sampling")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.dataset=="incar":
        from scripts.dataset_incar import EvalDataset
    elif args.dataset=="camrest":
        print("Evaluating on the dataset:",args.dataset)
        from scripts.dataset_camrest import EvalDataset
    elif args.dataset=="woz2_1":
        from scripts.dataset_multiwoz import EvalDataset


    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.top_weights = args.top_weights
    dataset_args.ir = args.ir
    dataset_args.topk = args.topk
    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.output_dir = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Generation parameters %s", args)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = EvalDataset(dataset_args, tokenizer, name=args.dataset, split_type=args.eval_dataset, labels_file=args.labels_file)
        print("---------Instantiated the test dataset---------------------")
        result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or args.eval_dataset)
    return result


if __name__ == "__main__":
    main()
