import argparse
import logging
import math
import os
import sys
import re
from collections import defaultdict
from logging.config import fileConfig

import numpy as np
import pandas as pd
import torch
from bert_score.scorer import BERTScorer
from sacrebleu import sentence_bleu
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append("../")
from approaches.run_approaches import count_keywords
from approaches.run_approaches import load_models_and_tokenizers, load_config
from finetune.cdg_dataloader import CDGDataLoader
from finetune import generation

IGNORE_INDEX = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("application.Evaluate")


def strip_resp(resp):
    if "<|endoftext|>" in resp:
        resp = resp.split("<|endoftext|>")[0].strip()
    if "__CUSTOMER__" in resp:
        resp = resp.split("__CUSTOMER__")[1].strip()
    return resp


def evaluate(config, model, tokenizer, output_dir, eval_filename=None, agent_model=None, agent_tokenizer=None):
    logger = logging.getLogger("application.Evaluation")
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    if eval_filename is None:
        num_eval_examples = int(config.get('TRAINING', 'num_eval_examples', fallback=None))
        eval_filename = f"{num_eval_examples}examples_eval_results.txt"

    eval_dataset = CDGDataLoader(config, model, tokenizer, split="eval")

    if os.path.isfile(config["TRAINING"]["eval_keywords_file_path"]):
        keywords = torch.load(config['TRAINING']['eval_keywords_file_path'])
    else:
        keywords = None

    assert config.get("DEFAULT", "train_task", fallback=None) is not None
    eval_task = config["DEFAULT"]["train_task"]

    assert config.get("DEFAULT", "local_rank", fallback=None) is not None

    local_rank = int(config["DEFAULT"]["local_rank"])

    assert config.get("TRAINING", "per_gpu_eval_batch_size", fallback=None) is not None
    per_gpu_eval_batch_size = int(config['TRAINING']['per_gpu_eval_batch_size'])

    no_cuda = config.getboolean("DEFAULT", "no_cuda", fallback=False)
    logger.info(f"No Cuda: {no_cuda}")

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        n_gpu = 1
        device = torch.device("cuda", local_rank)

    model = model.to(device)
    models = {"customer": model, "agent": agent_model}
    tokenizers = {"customer": tokenizer, "agent": agent_tokenizer}
    evaluation_metrics = config.get("TRAINING", "evaluation_metrics", fallback=None)
    # import pdb; pdb.set_trace()
    if evaluation_metrics is not None:
        evaluation_metrics = evaluation_metrics.split(",")
    else:
        raise ValueError("evaluation_metrics arg under the 'TRAINING' section of the config is not provided")

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info(f"***** Running evaluation {output_dir} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_results = defaultdict(list)
    if "bert-score" in evaluation_metrics:
        bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True)

    block_method = None
    if config.get("TRAINING", "block_method", fallback=None) is not None:
        block_method = config["TRAINING"]["block_method"]

    count = 0
    num_eval_examples = int(config.get('TRAINING', 'num_eval_examples', fallback=None))
    logger.info(f"NUM_EVAL_EXAMPLES {num_eval_examples}")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if count > num_eval_examples:
            continue
        with torch.no_grad():
            if block_method == "block-onehot":
                tag_vecs = batch[3].to(device)

            if config['TRAINING']['training_type'] == "regular":
                issueids, inputs, mask_customer, mask_agent = batch[:4]

            inputs = inputs.to(device)

            label_mask = mask_customer.new_zeros(mask_customer.size())
            if "customer" in eval_task:
                label_mask += mask_customer
            if "agent" in eval_task:
                label_mask += mask_agent
            if eval_task == "all":
                label_mask = inputs != tokenizer.pad_token_id
            label_mask = label_mask.to(device)
            labels = inputs.clone().detach()
            labels[label_mask == 0] = IGNORE_INDEX

            if block_method == "block-onehot":
                outputs = model(inputs, labels=labels, tag_vecs=tag_vecs)
            else:
                if config['TRAINING']['training_type'] == "regular":
                    outputs = model(inputs, labels=labels)

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                lm_loss = outputs[0]
            elif isinstance(outputs, dict):
                lm_loss = outputs['loss']
            else:
                logger.error(f"Outputs of type {type(outputs)} detected")
                assert False

            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

        conv_context = tokenizer.decode(inputs[0]).split("__PAD__")[0]
        if "__CUSTOMER__" in conv_context:
            conv_parts = [part.strip() for part in conv_context.split("<|endoftext|>") if part.strip() != ""]
            true_resp = conv_parts[-1]
            ctx = "".join([f"{part} <|endoftext|> " for part in conv_parts[:-1]])
            if "__CUSTOMER__" not in true_resp:
                print(f"Context: {ctx}")
                print(f"True Resp: {true_resp}")
                import pdb; pdb.set_trace()
        else:
            print(conv_context)
            continue

        # WARNING: SimpleDataloader should not go into this block
        # Some of the calculations in the following block needs keywords
        if any(item in evaluation_metrics for item in ['bleu-score', 'bert-score', 'success-rate']):
            generated_conv = generation.generate_all(ctx, models, tokenizers, num_convs=1, num_timesteps=1)[0]
            generated_resp = generated_conv.strip()
            generated_resp = strip_resp(generated_resp)

            generated_conv = "__CUSTOMER__ " + generated_resp
            if "success-rate" in evaluation_metrics and keywords is not None:
                # Success rate
                generation_context = conv_context.split("__PAST__")[1].split("__CURRENT__")[0]
                issueid_keywords = keywords[int(issueids[0])].split()
                past_cust_utts = " ".join(
                    [x.split("<|endoftext|>")[0] for x in generation_context.split("__CUSTOMER__")[1:]])
                past_cust_utts = re.sub('[^A-Za-z]+', ' ', past_cust_utts.lower().strip())
                future_cust_utts = " ".join(
                    [x.split("<|endoftext|>")[0] for x in generated_conv.split("__CUSTOMER__")[1:]])
                future_cust_utts = re.sub('[^A-Za-z]+', ' ', future_cust_utts.lower().strip())
                success_rate, past_keywords, future_keywords, generated_keywords = count_keywords(future_cust_utts,
                                                                                                  past_cust_utts,
                                                                                                  issueid_keywords)
                all_results['success-rate'].append(success_rate)
            if "bleu-score" in evaluation_metrics:
                bleu_score = sentence_bleu(generated_resp, [true_resp]).score
                all_results['bleu-score'].append(bleu_score)
            if "bert-score" in evaluation_metrics:
                P, R, F = bert_scorer.score([generated_resp], [true_resp])
                bert_score = max(F).item()
                all_results['bert-score'].append(bert_score)
        count += 1
    eval_loss = eval_loss / nb_eval_steps
    perplexity = math.exp(eval_loss)

    result = {}
    if "perplexity" in evaluation_metrics:
        result["loss"] = eval_loss
        result["perplexity"] = perplexity
    for r in all_results:
        result[r + '-mean'] = np.mean(all_results[r])
        result[r + '-std'] = np.std(all_results[r])

    logger.info(f"OUTPUT DIR: {output_dir}")
    with open(f"{output_dir}/{eval_filename}", "w") as writer:
        logger.info(f"***** Eval results {output_dir} *****")
        for key in sorted(result.keys()):
            logger.info(f" {key} = {str(result[key])}")
            writer.write(f" {key} = {str(result[key])}\n")
    return result


def main(config_file=None, config_obj=None):
    if config_obj is None:
        config = load_config(config_file)
    else:
        config = config_obj

    assert config is not None

    print("Running Eval using the following config")
    print(config)

    assert config.has_option("TRAINING", "output_dir")
    model_path = config["TRAINING"]["output_dir"]

    block_size = int(config['DATALOADER']['block_size'])

    num_eval_examples = int(config.get('TRAINING', 'num_eval_examples', fallback=None))
    eval_filename = f"{num_eval_examples}examples_eval_results.txt"
    if config['TRAINING'].getboolean('evaluate_all_checkpoints', fallback=False):
        checkpoints = [(int(c.split("checkpoint-")[1]), f"{model_path}/{c}") for c in os.listdir(model_path) if
                       "checkpoint-" in c]
        checkpoints = [c[1] for c in sorted(checkpoints, key=lambda x: x[0])]
        checkpoints = [c for c in checkpoints if not os.path.exists(os.path.join(c, eval_filename))]
    else:
        checkpoints = [f"{model_path}"]
    logger.info(f"CHECKPOINTS: {checkpoints}")
    if config.get("TRAINING", "agent_model_path", fallback=None) is not None:
        agent_model = GPT2LMHeadModel.from_pretrained(config['TRAINING']['agent_model_path'])
        agent_tokenizer = GPT2Tokenizer.from_pretrained(config['TRAINING']['agent_model_path'])
    else:
        agent_model = None
        agent_tokenizer = None
    results = {}
    for checkpoint in checkpoints:
        logger.info(f"EVALUATING CHECKPOINT: {checkpoint}")
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model, tokenizer = load_models_and_tokenizers(config, task="eval", checkpoint_path=checkpoint)
        if block_size <= 0:
            block_size = (
                tokenizer.max_len_single_sentence
            )  # Our input block size will be the max possible for the model

        result = evaluate(config, model, tokenizer, checkpoint, eval_filename, agent_model, agent_tokenizer)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)

    args = parser.parse_args()

    fileConfig("../logger_config.conf")

    main(args.config_file)
