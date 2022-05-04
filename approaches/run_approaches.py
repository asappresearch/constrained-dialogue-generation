import argparse
import itertools
import logging
import math
import os
import sys
import random
import re
import numpy as np
from collections import defaultdict
from logging.config import fileConfig
from timeit import default_timer as timer

import pandas as pd
import torch
from bert_score.scorer import BERTScorer
from nltk.stem import PorterStemmer
from sacrebleu import sentence_bleu
from tqdm import tqdm

sys.path.append("../")
from datastore.knn_datastore import preprocess_conv
from approaches.encoders import GPT2Encoder
from approaches.cgmh.key_gen import cgmh_generate
from approaches.helpers import get_preprocessed_cust_utt, get_conv_ngrams, get_tfidf_vectorizer, get_tfidf_topn, get_cust_utts_future, get_cust_utts_past, count_keywords, set_seed
from approaches.loader import load_models_and_tokenizers, load_config, load_datastore, load_data
from approaches.dbs import conditional_language_generation as dbs
from approaches.window_fop import conditional_language_generation as window_fop


porter = PorterStemmer()
seed=1234
set_seed(seed)
NOCUDA = os.getenv('NOCUDA', False)
DEVICE = "cuda" if torch.cuda.is_available() and not NOCUDA else "cpu"
bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True)
logger = logging.getLogger("application.Main")


class CGMH:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        try:
            logger.info(f'CONTEXT: {context}')
            logger.info(f'KEYWORDS: {control_words_to_generate}')
            return cgmh_generate(context, control_words_to_generate, models[speaker], tokenizers[speaker])
        except Exception as e:
            logger.error(e)
            if 'out of memory' in str(e):
                try:
                    logger.info(f'CONTEXT: {context}')
                    logger.info(f'KEYWORDS: {control_words_to_generate}')
                    logger.info('Trying with search size of 50 due to OOM')
                    torch.cuda.empty_cache()
                    return cgmh_generate(context, control_words_to_generate, models[speaker], tokenizers[speaker],
                                         search_size=50)
                except Exception as e:
                    logger.error(e)
                    torch.cuda.empty_cache()
                    return None


class Retrieval:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        return knn_resp


class PromptingWithTraining:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        context = "__KEYWORDS__ " + " ".join(control_words_to_generate) + " " + context
        generated_resp = model_generate(context, None, models, tokenizers, speaker, args.num_tokens_to_produce)
        return generated_resp


class Prompting:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        context = " ".join(control_words_to_generate) + " " + context
        generated_resp = model_generate(context, None, models, tokenizers, speaker, args.num_tokens_to_produce)
        return generated_resp


class DirectedBeamSearch:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        return run_dbs(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                       speaker='customer', use_knn=False, approach="dbs")


class WindowFuturesOfThePast:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        return run_fop(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp, speaker,
                       use_knn=True, approach="window_fop", window_size=4, upweight_control_words=False)


class FuturesOfThePast:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        return run_fop(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp, speaker,
                       use_knn=True, approach="window_fop", window_size=1, upweight_control_words=False)


class WFirst:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        print(control_words_to_generate)
        return " ".join(control_words_to_generate)


class FinetunedModel:
    def generate(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp,
                 speaker='customer', orig_control_words=None):
        generated_resp = model_generate(context, None, models, tokenizers, speaker, args.num_tokens_to_produce)
        return generated_resp


def run_dbs(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp, speaker, use_knn,
            approach):
    tmp_save_dir = f"{args.save_dir}/{args.eval_type}_{args.condition.lower()}_{args.num_keywords}keywords"
    os.makedirs(tmp_save_dir, exist_ok=True)
    guide_words = None
    if use_knn:
        processed_knn_resp = re.sub('[^A-Za-z]+', ' ', knn_resp.lower().strip())
        guide_words = processed_knn_resp.split()
    else:
        guide_words = control_words_to_generate
    logger.info(f"KEYWORDS TO GENERATE: {guide_words}")
    evaluation = eval(approach)(models[speaker], tokenizers[speaker],
                                seed=seed,
                                word_index=example_index,
                                number_of_concurrent_sentences=args.number_of_concurrent_sentences,
                                number_of_generated_sentences=args.number_of_generated_sentences,
                                number_of_words_per_sentence=args.num_tokens_to_produce,
                                number_of_beams=args.number_of_beams,
                                final_text_file=f"{tmp_save_dir}/final_text.txt",
                                num_keywords=len(guide_words),
                                constant=args.lambda_param, embedding_type="glove",
                                save_path=tmp_save_dir,
                                index_keywords=guide_words,
                                agent_model_path=None,
                                first_utt=context)
    generated_resp = evaluation['generated_future: ']
    if "__CUSTOMER__" in generated_resp:
        generated_resp = generated_resp.split("__CUSTOMER__")[1]
    if "<|endoftext|>" in generated_resp:
        generated_resp = generated_resp.split("<|endoftext|>")[0].strip()
    return generated_resp


def run_fop(args, example_index, models, tokenizers, context, control_words_to_generate, knn_resp, speaker, use_knn,
            approach, window_size, upweight_control_words=False):
    tmp_save_dir = f"{args.save_dir}/{args.eval_type}_{args.condition.lower()}_{args.num_keywords}keywords"
    os.makedirs(tmp_save_dir, exist_ok=True)
    guide_words = None
    if use_knn:
        processed_knn_resp = re.sub('[^A-Za-z]+', ' ', knn_resp.lower().strip())
        guide_words = processed_knn_resp.split()
    else:
        guide_words = control_words_to_generate
    logger.info(f"KEYWORDS TO GENERATE: {guide_words}")
    tmp_df = pd.DataFrame()
    scored_resp_list = []
    for i in range(args.num_candidate_generations):
        evaluation = eval(approach)(models[speaker], tokenizers[speaker],
                                    seed=seed,
                                    word_index=example_index,
                                    number_of_concurrent_sentences=1,
                                    number_of_generated_sentences=1,
                                    number_of_words_per_sentence=args.num_tokens_to_produce,
                                    number_of_beams=1,
                                    final_text_file=f"{tmp_save_dir}/final_text.txt",
                                    num_keywords=len(guide_words),
                                    constant=args.lambda_param, embedding_type="glove",
                                    save_path=tmp_save_dir,
                                    index_keywords=guide_words,
                                    agent_model_path=None,
                                    first_utt=context,
                                    window_size=window_size,
                                    orig_control_words=control_words_to_generate,
                                    upweight_control_words=upweight_control_words,
                                    debug=args.debug)
        generated_resp = evaluation['generated_future: ']
        if "__CUSTOMER__" in generated_resp:
            generated_resp = generated_resp.split("__CUSTOMER__")[1]
        if "<|endoftext|>" in generated_resp:
            generated_resp = generated_resp.split("<|endoftext|>")[0].strip()

        logger.info(f"CONTROL WORDS: {control_words_to_generate}")
        logger.info(f"KNN RESP: {knn_resp}")
        logger.info(f"GENERATED RESP: {generated_resp}")
        loss, perplexity = compute_loss_ppl(models, tokenizers, context, generated_resp)
        success_rate = compute_success_rate(context, generated_resp, control_words_to_generate)
        score = success_rate
        new_row = {"example_index": example_index, "approach": approach,
                   "control_words_to_generate": control_words_to_generate, "knn_resp": knn_resp,
                   "generated_resp": generated_resp, "loss": loss, "success-rate": success_rate}
        tmp_df = tmp_df.append(new_row, ignore_index=True)
        scored_resp_list.append((generated_resp, score, success_rate, loss))
        logger.info(f"RESULTS: index={i}, success-rate={success_rate}, loss={loss}\nGenerated-resp={generated_resp}")

    sorted_scored_resp_list = sorted(scored_resp_list, key=lambda x: -x[1])
    best_score = sorted_scored_resp_list[0][1]
    high_success_rate_resps = [sorted_scored_resp_list[p] for p in range(len(sorted_scored_resp_list)) if
                               sorted_scored_resp_list[p][1] == best_score]
    sorted_loss_list = sorted(high_success_rate_resps, key=lambda x: x[3])
    generated_resp = sorted_loss_list[0][0]
    logger.info(f"FINAL generated resp: {generated_resp}")
    return generated_resp


def add_endoftext_tokens(s, endoftext_token="<|endoftext|>"):
    new_s = ""
    for word in s.split():
        if (word == "__CUSTOMER__" or word == "__AGENT__") and len(new_s) > 0:
            if new_s.strip().split(endoftext_token)[-1] != '':
                new_s += endoftext_token + " "
        new_s += word + " "

    if new_s.strip().split(endoftext_token)[-1] != '':
        new_s += endoftext_token

    return new_s.strip()


def retrieve_futures(split, example_index, num_neighbors, num_futures, data, train_data, curr_context_data,
                     train_context_data, curr_dstore, train_dstore, future_type, issueid_keywords, fp16=True,
                     hidden_states=None):  # future_type = ['good', 'bad']
    if hidden_states is None:
        past_contexts = list(curr_dstore.keys)
        future_contexts = list(curr_dstore.vals)
        past_context = past_contexts[example_index]
    else:
        past_context = hidden_states
    _, knns = train_dstore.get_knns(past_context.reshape((1, 1024)))
    knns = knns[0]
    past_issueid, past_turn_num = curr_context_data.loc[example_index]['issue_ids'], \
                                  curr_context_data.loc[example_index]['current_turn_idx']
    past_example = data[past_issueid]

    knn_keywords = []
    for k in knns:
        if k == -1:
            continue
        knn_issueid, knn_turn_num = train_context_data.loc[k]['issue_ids'], train_context_data.loc[k][
            'current_turn_idx']
        if past_issueid == knn_issueid:
            continue

        true_past_text = get_cust_utts_past(past_example, past_turn_num)
        knn_future_text = get_cust_utts_future(train_data[knn_issueid], knn_turn_num)
        num_keywords, _, _, _ = count_keywords(knn_future_text, true_past_text, issueid_keywords)
        knn_keywords.append((k, num_keywords))
    if future_type == "good":
        sorted_knn_keywords = sorted(knn_keywords, key=lambda x: -x[1])
    elif future_type == "bad":
        sorted_knn_keywords = sorted(knn_keywords, key=lambda x: x[1])
    best_nns = []
    for n in sorted_knn_keywords:
        if n[0] not in best_nns and len(best_nns) < num_futures:
            best_nns.append(n[0])
    return best_nns


def retrieve_knn_resp(args, info, control_words_to_generate, hidden_states):
    knn_resp = None
    if args.condition in ['Retrieval', 'FuturesOfThePast',
                          'RetrievalDBS'] or 'Retrieval' in args.condition or 'FuturesOfThePast' in args.condition:
        best_nns = retrieve_futures("eval", info['e_index'], 1024, args.num_futures, info['preprocess_data'],
                                    info['train_data'], info['curr_context_data'], info['train_context_data'],
                                    info['curr_dstore'], info['train_dstore'], "good", control_words_to_generate,
                                    hidden_states=hidden_states)
        knn_resp_list = []
        for best_nn in best_nns:
            past_nn_text = info['train_context_data']['contexts'].loc[best_nn]
            future_nn_text = info['train_context_data']['future'].loc[best_nn]
            knn_resp = ""
            if "__CUSTOMER__" in future_nn_text:
                knn_resp = future_nn_text.split("__CUSTOMER__")[1]
            if "__" in knn_resp:
                knn_resp = knn_resp.split("__")[0].strip()
            knn_resp_list.append(knn_resp)
        return knn_resp_list[0]
    return knn_resp


def compute_success_rate(context, generated_resp, control_words_to_generate):
    # Compute success rate
    past_cust_utts = " ".join([x.split("<|endoftext|>")[0] for x in context.split("__CUSTOMER__")[1:]])
    past_cust_utts = re.sub('[^A-Za-z]+', ' ', past_cust_utts.lower().strip())
    future_cust_utts = generated_resp
    future_cust_utts = re.sub('[^A-Za-z]+', ' ', future_cust_utts.lower().strip())
    success_rate, past_keywords, future_keywords, generated_keywords = count_keywords(future_cust_utts, past_cust_utts,
                                                                                      control_words_to_generate)
    return success_rate


def compute_loss_ppl(models, tokenizers, context, generated_resp):
    try:
        endoftext_token = "<|endoftext|>"
        if generated_resp.strip().split(endoftext_token)[-1] != '':
            generated_resp += endoftext_token
        context_input_ids = tokenizers['customer'].encode(context)
        generated_input_ids = tokenizers['customer'].encode(generated_resp)
        indexed_tokens = context_input_ids + generated_input_ids
        indexed_labels = [-100] * len(context_input_ids) + generated_input_ids
        max_len = 1024
        if len(indexed_tokens) > max_len:
            model_indexed_tokens = indexed_tokens[-max_len:]
            labels = indexed_labels[-max_len:]
        else:
            model_indexed_tokens = indexed_tokens
            labels = indexed_labels

        assert len(model_indexed_tokens) == len(labels)
        input_ids = torch.tensor([model_indexed_tokens]).to(DEVICE)
        labels = torch.tensor([labels]).to(DEVICE)
        model_output = models['customer'](input_ids, labels=labels)
        loss = model_output['loss'].item()
        perplexity = math.exp(loss)
    except:
        loss, perplexity = 100000, 100000
    print("LOSS: ", loss, " PERPLEXITY: ", perplexity)
    return loss, perplexity


def compute_eval_metrics(args, method, info, models, tokenizers, context, true_resp, generated_resp, orig_control_words,
                         num_tokens_to_produce):
    # Compute true and generated words
    _, _, _, true_resp_keywords = count_keywords(true_resp, "", orig_control_words)
    _, _, _, generated_resp_keywords = count_keywords(generated_resp, "", orig_control_words)
    true_resp_keywords = set(true_resp_keywords)
    generated_resp_keywords = set(generated_resp_keywords)
    true_positives = true_resp_keywords.intersection(generated_resp_keywords)

    logger.info(f"Context: {context}")
    logger.info(f"Control words: {orig_control_words}")
    logger.info(f"True resp: {true_resp} Keywords: {true_resp_keywords}")
    logger.info(f"Generated resp: {generated_resp} Keywords: {generated_resp_keywords}")
    logger.info(f"True positives: {true_positives}")

    # Compute BLEU score
    bleu_score = sentence_bleu(generated_resp, [true_resp]).score

    # Compute BERT score
    P, R, F = bert_scorer.score([generated_resp], [true_resp])
    bert_score = max(F).item()

    # Compute loss and success-rate
    loss, perplexity = compute_loss_ppl(models, tokenizers, context, generated_resp)
    success_rate = compute_success_rate(context, generated_resp, orig_control_words)

    curr_results = {'success-rate': success_rate, 'BLEU-score': bleu_score, 'BERT-score': bert_score,
                    'true_resp_keywords': " ".join(true_resp_keywords),
                    'generated_resp_keywords': " ".join(generated_resp_keywords),
                    'true+generated_keywords': " ".join(true_positives), 'loss': loss, 'perplexity': perplexity}
    return curr_results


def model_generate(context, input_ids, models, tokenizers, speaker, num_tokens_to_produce, p=0.5, ngram_block=4):
    if input_ids is None:
        indexed_tokens = tokenizers[speaker].encode(context)
        max_len = 1024
        if len(indexed_tokens) > max_len:
            model_indexed_tokens = indexed_tokens[-max_len:]
        else:
            model_indexed_tokens = indexed_tokens
        input_ids = torch.tensor([model_indexed_tokens]).to(DEVICE)

    model_type = "GPT2"
    block_size = 512 if model_type == "pegasus" else 1024
    bad_words_ids = [
        tokenizers[speaker].encode("__PAST__"),
        tokenizers[speaker].encode("__CURRENT__"),
        tokenizers[speaker].encode("__FUTURE__"),
        tokenizers[speaker].encode("<mask_1>"),
        tokenizers[speaker].encode("__KEYWORDS__"),
        tokenizers[speaker].encode("__PAD__"),
        tokenizers[speaker].encode("__AGENT__"),
        tokenizers[speaker].encode("__CUSTOMER__")]
    out = models[speaker].generate(
        input_ids=input_ids,
        min_length=input_ids.size(1) + 2,
        max_length=num_tokens_to_produce + input_ids.size(1),
        top_p=p,
        do_sample=True,
        no_repeat_ngram_size=ngram_block,
        pad_token_id=tokenizers[speaker].pad_token_id,
        eos_token_id=tokenizers[speaker].eos_token_id,
        forced_eos_token_id=tokenizers[speaker].eos_token_id,
        bad_words_ids=bad_words_ids)
    out = out[:, input_ids.size(1):]
    generated_output = tokenizers[speaker].batch_decode(out)[0]
    if "__" + speaker.upper() + "__" in generated_output:
        generated_output = generated_output.split("__" + speaker.upper() + "__")[1]
    if "<|endoftext|>" in generated_output:
        generated_output = generated_output.split("<|endoftext|>")[0]
    generated_resp = generated_output
    return generated_resp


def get_all_conv_keywords(starting_keywords, context, true_future, num_keywords):
    # Compute keywords for each conversation
    future_cust_utts = [x.split("<|endoftext|>")[0] for x in true_future.split("__CUSTOMER__")[1:]]
    future_cust_utts = [re.sub('[^A-Za-z]+', ' ', u.lower().strip()) for u in future_cust_utts]
    utt_keywords = [[word for word in f.split() if word in starting_keywords] for f in future_cust_utts]
    if sum([len(u) for u in utt_keywords]) < num_keywords:
        return None
    curr_utt_index = 0
    final_keywords = defaultdict(list)
    while sum([len(final_keywords[f]) for f in final_keywords]) < num_keywords:
        if len(final_keywords[curr_utt_index]) < len(utt_keywords[curr_utt_index]):
            next_word = utt_keywords[curr_utt_index][len(final_keywords[curr_utt_index])]
            final_keywords[curr_utt_index].append(next_word)
        curr_utt_index += 1
        if curr_utt_index >= len(utt_keywords):
            curr_utt_index = 0
    control_words_to_generate = [final_keywords[i] for i in final_keywords]
    control_words_to_generate = list(itertools.chain(*control_words_to_generate))
    logger.info(f"UTT KEYWORDS: {utt_keywords}")
    logger.info(f"BEFORE CONTROL WORDS TO GENERATE: {control_words_to_generate}")
    random.shuffle(control_words_to_generate)
    logger.info(f"CONTROL WORDS TO GENERATE: {control_words_to_generate}")
    return control_words_to_generate


def load_all(args):
    # Load models, tokenizers, keywords, etc
    config = load_config(args.config_file)
    if args.condition == "PromptingWithTraining":
        model, tokenizer = load_models_and_tokenizers(config, "eval", checkpoint_path=args.prompting_model_path)
    else:
        model, tokenizer = load_models_and_tokenizers(config, "eval", checkpoint_path=args.model_path)
    model = model.to(DEVICE)
    logger.info(f"MODEL_PATH: {args.model_path} LEN TOKENIZER: {len(tokenizer)} "
                f"TOKENIZER ADDED VOCAB: {tokenizer.get_added_vocab()}")
    models = {"customer": model}
    tokenizers = {"customer": tokenizer}

    if not args.one_step_generation:
        assert os.path.exists(args.agent_model_path)
        agent_model, agent_tokenizer = load_models_and_tokenizers(config, "eval", checkpoint_path=args.agent_model_path)
        agent_model = agent_model.to(DEVICE)
        models["agent"] = agent_model
        tokenizers["agent"] = agent_tokenizer
    return config, models, tokenizers


def remove_generated_words(control_words, generated_utts):
    if len(generated_utts) == 0:
        return control_words
    already_generated_word_indices = []
    generated_utts = generated_utts.split()
    generated_stems = [porter.stem(generated_utts[i]).lower().strip() for i in range(len(generated_utts))]
    for c in range(len(control_words)):
        word = control_words[c]
        word_stem = porter.stem(word).lower().strip()
        if word_stem in generated_stems:
            already_generated_word_indices.append(c)
    return [control_words[c] for c in range(len(control_words)) if c not in already_generated_word_indices]


def find_keywords(config, split, keyword_type="conversation-tfidf"):
    raw_train_data = load_data(config['DATALOADER']["train_data_file"], "train")
    eval_data = load_data(config['DATALOADER'][f"{split}_data_file"], split)
    dataloader_config = config["DATALOADER"]

    data = {issueid: get_preprocessed_cust_utt(eval_data[issueid]) for issueid in tqdm(eval_data)}
    train_data = {issueid: get_preprocessed_cust_utt(raw_train_data[issueid]) for issueid in raw_train_data}

    keywords_dir = config.get("DATALOADER", "keywords_dir", fallback=None)
    assert keywords_dir is not None

    if not os.path.exists(keywords_dir):
        os.makedirs(keywords_dir, exist_ok=True)

    # Get keywords depending on prompt_type
    if keyword_type == "conversation-tfidf":
        dest_file = os.path.join(keywords_dir, f"eval_{keyword_type}.pkl")
        if os.path.isfile(dest_file):
            return

        corpus = [" ".join(train_data[issueid]) for issueid in train_data]
        tfidf_vectorizer = get_tfidf_vectorizer(corpus, min_df=1)

        keywords = {issueid: get_tfidf_topn(" ".join(data[issueid]), tfidf_vectorizer) for issueid in tqdm(data)}
        torch.save(keywords, dest_file)
    else:
        assert False
    return keywords


def run_approach(args):
    if args.condition == "PromptingWithTraining":
        assert args.prompting_model_path is not None
    os.makedirs(args.save_dir, exist_ok=True)
    config, models, tokenizers = load_all(args)

    assert args.num_eval_examples <= args.num_total_examples
    if args.keywords_file_path is None:
        args.keywords_file_path = config["TRAINING"]["eval_keywords_file_path"]

    # Filename + dataframe to save all results
    if args.condition == "PromptingWithTraining":
        prompt_dir, prompt_checkpoint = os.path.split(args.prompting_model_path)
        eval_df_filename = f"{args.save_dir}/{args.eval_type.upper()}_{args.condition}_{args.lambda_param}lambda_{args.num_keywords}keywords_{args.num_futures}futures_{args.num_candidate_generations}generations_{args.num_total_examples}examples_prompt{prompt_checkpoint}.csv"
        print("PROMPT eval df filename: ", eval_df_filename)
    elif args.percentdata_datastore is not None:
        datastore_dir, datastore_name = os.path.split(args.percentdata_datastore)
        eval_df_filename = f"{args.save_dir}/{args.eval_type.upper()}_{args.condition}_{args.lambda_param}lambda_{args.num_keywords}keywords_{args.num_futures}futures_{args.num_candidate_generations}generations_{args.num_total_examples}examples_{datastore_name}.csv"
    else:
        eval_df_filename = f"{args.save_dir}/{args.eval_type.upper()}_{args.condition}_{args.lambda_param}lambda_{args.num_keywords}keywords_{args.num_futures}futures_{args.num_candidate_generations}generations_{args.num_total_examples}examples.csv"

    last_eval_index = -1
    eval_df = pd.DataFrame()
    if os.path.exists(eval_df_filename):  # Skip if this setting has already been run
        try:
            eval_df = pd.read_csv(eval_df_filename)
            last_eval_index = int(
                max(eval_df['example_index']))  # Get the last example index that was already run to not redo those
        except:
            return  # If example index is not in the DataFrame, it might be an old file, which did not have intermediate saving. So skip this experiment if this is the case.

    # Load keywords, training preprocess_data, and knn datastore
    split = "test" if args.split == "eval" else args.split
    if args.percentdata_datastore is None:
        datastore_path = config['DATALOADER']['datastore_file']
        train_data_path = config['DATALOADER']['train_data_file']
    else:
        assert args.percentdata_datafile is not None
        datastore_path = args.percentdata_datastore  # f"preprocess_data/ABCD/varying_data/abcd_{args.percent_historical_data}percent_datastore"
        train_data_path = args.percentdata_datafile

    keywords_filename = f"{args.data_dir}/shuffled_{args.split}split_{args.num_keywords}keywords.pkl"
    curr_context_data = pd.read_csv(os.path.join(config['DATALOADER']['datastore_file'], f"context_data_{split}.csv"))
    train_context_data = pd.read_csv(os.path.join(datastore_path, "context_data_train.csv"))
    data = load_data(config['DATALOADER'][args.split + '_data_file'], args.split)
    train_data = load_data(train_data_path, "train")

    if args.percentdata_datastore is None:
        n_test_datastore_keys = int(config["DATALOADER"][f"kvstore_n_{split}contexts"])
        n_train_datastore_keys = int(config["DATALOADER"]["kvstore_n_traincontexts"])
    else:
        n_test_datastore_keys = len(curr_context_data)
        n_train_datastore_keys = len(train_context_data)

    curr_dstore = load_datastore(config['DATALOADER']['datastore_file'], args.num_neighbors, args.split,
                                 n_test_datastore_keys, fp16=args.fp16)
    train_dstore = load_datastore(datastore_path, args.num_neighbors, "train", n_train_datastore_keys, fp16=args.fp16)

    # Generating control words for each conversation
    if not os.path.exists(keywords_filename):
        eval_issueids, eval_conv_and_keywords = [], {}

        if os.path.isfile(args.keywords_file_path):
            all_keywords = torch.load(args.keywords_file_path)
        else:
            all_keywords = find_keywords(config, split=args.split)

        for e_index in tqdm(range(len(curr_context_data))):
            issueid, turn_num = curr_context_data.loc[e_index]['issue_ids'], curr_context_data.loc[e_index][
                'current_turn_idx']
            if issueid in eval_issueids:  # First generate control words for each issueid
                continue
            eval_issueids.append(issueid)
            logger.info(f"EXAMPLE INDEX: {e_index} ISSUEID: {issueid} NUM ISSUEIDS: {len(eval_issueids)}")
            context = add_endoftext_tokens(curr_context_data['contexts'].loc[e_index]) + " __CUSTOMER__ "
            true_future = add_endoftext_tokens(curr_context_data['future'].loc[e_index])

            control_words_to_generate = get_all_conv_keywords(all_keywords[issueid].split(), context, true_future,
                                                              args.num_keywords)
            if control_words_to_generate is None:  # Skip if there aren't enough control words in this conversation
                continue
            eval_conv_and_keywords[issueid] = (e_index, control_words_to_generate)
        torch.save(eval_conv_and_keywords, keywords_filename)
    else:
        eval_conv_and_keywords = torch.load(keywords_filename)

    # Generate with each approach + evaluate
    method = eval(args.condition)

    for e_index in tqdm(
            range(last_eval_index + 1, len(curr_context_data))):  # Start from where the previous results ended
        context = add_endoftext_tokens(curr_context_data['contexts'].loc[e_index]) + " __CUSTOMER__ "
        true_future = add_endoftext_tokens(curr_context_data['future'].loc[e_index])
        issueid, turn_num = curr_context_data.loc[e_index]['issue_ids'], curr_context_data.loc[e_index][
            'current_turn_idx']
        if e_index > args.num_eval_examples or \
                issueid not in eval_conv_and_keywords or \
                "__CUSTOMER__" not in true_future or \
                (
                        args.limit_turns_to is not None and turn_num > args.limit_turns_to):  # Skip if you've already run for the desired number of examples, you don't have keywords for this conversation, or if there's no future customer utterance left
            continue
        saved_example_index, orig_control_words = eval_conv_and_keywords[issueid]
        if args.eval_type == "real" and len(eval_df) > 0:
            tmp_df = eval_df[eval_df['issueid'] == issueid]
            if len(tmp_df) > 0:
                tmp_df.fillna('', inplace=True)
                previously_generated_words = tmp_df.groupby('issueid')['generated_resp_keywords'].apply(" ".join).iloc[
                    0].split()
                control_words_to_generate = []
                for w in orig_control_words:
                    if w not in previously_generated_words:
                        control_words_to_generate.append(w)
            else:
                control_words_to_generate = orig_control_words
        else:
            control_words_to_generate = orig_control_words

        info = {'e_index': e_index, 'preprocess_data': data, 'train_data': train_data,
                'curr_context_data': curr_context_data, 'train_context_data': train_context_data,
                'curr_dstore': curr_dstore, 'train_dstore': train_dstore}
        # Remove control words that have already been used in the past
        past_text = " ".join([x.split("<|endoftext|>")[0] for x in context.split("__CUSTOMER__")[1:]])
        past_text = re.sub('[^A-Za-z]+', ' ', past_text.lower().strip())
        # context = re.sub('\s+', ' ', context)
        context = re.sub('\s+', ' ', context)

        new_row, curr_results, new_simulated_row = {}, {}, {}
        if args.eval_type in ['real']:
            # Get true response
            true_resp = true_future.split("<|endoftext|>")[0].split("__CUSTOMER__")[1].strip()
            # Retrieve relevant futures for relevant conditions
            knn_resp = retrieve_knn_resp(args, info, control_words_to_generate, None)
            # Run approach's generate function
            generated_resp = method.generate(args, e_index, models, tokenizers, context, control_words_to_generate,
                                             knn_resp, speaker='customer', orig_control_words=orig_control_words)

            if generated_resp is None:
                generated_resp = ""
                # continue
            _, _, _, generated_keywords = count_keywords(generated_resp, '', control_words_to_generate)
            # Compute evaluation metrics
            curr_results = compute_eval_metrics(args, method, info, models, tokenizers, context, true_resp,
                                                generated_resp, orig_control_words, args.num_tokens_to_produce)
            # Save all results and examples into DataFrame
            new_row = {'example_index': int(e_index), 'issueid': int(issueid), 'turn_num': int(turn_num),
                       'lambda': args.lambda_param, 'context': context, 'true_resp': true_resp, 'knn_resp': knn_resp,
                       'generated_resp': generated_resp,
                       'control_words_to_generate': " ".join(control_words_to_generate),
                       "orig_control_words": " ".join(orig_control_words)}

            # Print qualitative results
            logger.info(f"CONDITION: {args.condition} LAMBDA: {args.lambda_param}")
            logger.info(f"Context: {context}")
            logger.info(f"True: {true_resp}")
            logger.info(f"KNN: {knn_resp}")
            logger.info(f"Generated: {generated_resp}")
            new_row = {**new_row, **curr_results}
            eval_df = eval_df.append(new_row, ignore_index=True)

        if args.eval_type in ['simulated']:
            encoder_model = GPT2Encoder.from_pretrained(args.model_path).to(DEVICE)

            if args.percentdata_datastore is None:
                assert args.datastore_for_simulation is not None
                curr_dstore = load_datastore(args.datastore_for_simulation,
                                             args.num_neighbors,
                                             args.split,
                                             n_test_datastore_keys,
                                             fp16=args.fp16)  # len(curr_context_data))
                train_dstore = load_datastore(args.datastore_for_simulation,
                                              args.num_neighbors,
                                              "train",
                                              n_train_datastore_keys,
                                              fp16=args.fp16)  # len(train_context_data))

            info = {'e_index': e_index,
                    'preprocess_data': data,
                    'train_data': train_data,
                    'curr_context_data': curr_context_data,
                    'train_context_data': train_context_data,
                    'curr_dstore': curr_dstore,
                    'train_dstore': train_dstore}
            if saved_example_index != e_index:  # Skip intermediate conversation responses. Only evaluate once per issueid/conversation
                continue
            # Simulate and compute success rate
            # Remove the last __CUSTOMER__ tag as it's not included in the datastore keys
            tmp_ctx = " ".join(context.split()[0:-1])
            nxt_cust_utterances = []
            for step_num in range(10):
                control_words_to_generate = remove_generated_words(control_words_to_generate, " ".join(
                    nxt_cust_utterances))  # Remove words already generated
                with torch.no_grad():
                    tmp_ctx_input_ids, tmp_attn_mask, tmp_position_ids = preprocess_conv(tokenizers['customer'],
                                                                                         [tmp_ctx])
                    hidden_states = encoder_model(
                        tmp_ctx_input_ids,
                        attention_mask=tmp_attn_mask,
                        position_ids=tmp_position_ids,
                        return_dict=True,
                    ).detach().cpu()
                    del tmp_ctx_input_ids, tmp_attn_mask, tmp_position_ids
                knn_resp = retrieve_knn_resp(args, info, control_words_to_generate, hidden_states)
                try:
                    tmp_ctx += " __CUSTOMER__ "
                    simulation_index = int(
                        str(e_index) + str(step_num) + str(args.num_keywords) + str(args.lambda_param))
                    print("SIMULATION INDEX: ", simulation_index)
                    if args.condition == "PromptingWithTraining":
                        nxt_cust_utt = method.generate(args, simulation_index, models, tokenizers, tmp_ctx,
                                                       orig_control_words, knn_resp,
                                                       orig_control_words=orig_control_words)
                    else:
                        nxt_cust_utt = method.generate(args, simulation_index, models, tokenizers, tmp_ctx,
                                                       control_words_to_generate, knn_resp,
                                                       orig_control_words=orig_control_words)
                    logger.info(f"CUSTOMER: {nxt_cust_utt}")

                    nxt_cust_utt = nxt_cust_utt[0:nxt_cust_utt.index(
                        "<|endoftext|>")].strip() if "<|endoftext|>" in nxt_cust_utt else nxt_cust_utt.strip()
                    processed_cust_utt = re.sub('[^A-Za-z]+', ' ', nxt_cust_utt.lower().strip())
                    nxt_cust_utterances.append(processed_cust_utt)

                    tmp_ctx += nxt_cust_utt + " <|endoftext|> __AGENT__ "
                    tmp_ctx_input_ids = torch.tensor(tokenizers['agent'].encode(tmp_ctx)).unsqueeze(0).to(DEVICE)
                    nxt_agent_utt = model_generate(tmp_ctx, tmp_ctx_input_ids, models, tokenizers, 'agent',
                                                   args.num_tokens_to_produce)
                    if not nxt_agent_utt:
                        break
                    logger.info(f"AGENT: {nxt_agent_utt}")
                    nxt_agent_utt = nxt_agent_utt[0:nxt_agent_utt.index(
                        "<|endoftext|>")].strip() if "<|endoftext|>" in nxt_agent_utt else nxt_agent_utt.strip()
                    tmp_ctx += nxt_agent_utt + " <|endoftext|>"
                except:
                    break  # For approaches like CGMH that may have an out-of-memory error with large contexts, break out of the rollout if this occurs
                logger.info(f"CONTEXT: {tmp_ctx}")

            success_rate, past_keywords, future_keywords, generated_keywords = count_keywords(
                " ".join(nxt_cust_utterances), past_text, orig_control_words)
            logger.info(f"Long-term success-rate: {success_rate}")

            true_conversation = " ".join(context.split()[0:-1]) + " " + true_future
            new_row = {'example_index': int(e_index),
                       'issueid': int(issueid),
                       'turn_num': int(turn_num),
                       'lambda': args.lambda_param,
                       'context': context,
                       'control_words_to_generate': " ".join(orig_control_words),
                       'true_conversation': true_conversation,
                       'simulated_conversation': tmp_ctx,
                       'long-term-success-rate': success_rate,
                       'past_keywords': " ".join(past_keywords),
                       'future_keywords': " ".join(future_keywords),
                       'generated_keywords': " ".join(generated_keywords)}
            eval_df = eval_df.append(new_row, ignore_index=True)

        if len(eval_df) > 0:
            eval_df.to_csv(eval_df_filename, index=False)
        # torch.save(already_generated_keywords, f"{args.save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General + experiment-specific arguments
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--eval_type", type=str, choices=['real', 'simulated'])
    parser.add_argument("--agent_model_path", type=str)
    parser.add_argument("--keywords_file_path", type=str, default=None)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--condition",
                        choices=['Retrieval', 'Prompting', 'DirectedBeamSearch', 'FuturesOfThePast', 'CGMH',
                                 'WindowFuturesOfThePast', 'WFirst', 'FinetunedModel', 'PromptingWithTraining'])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--num_eval_examples", type=int)
    parser.add_argument("--num_total_examples", type=int, default=100)
    parser.add_argument("--num_keywords", type=int)
    parser.add_argument("--num_tokens_to_produce", default=100, type=int)
    parser.add_argument("--split", default="eval", type=str)
    parser.add_argument("--one-step-generation", action="store_true",
                        help="flag to indicate only generate the next immediate response")
    parser.add_argument("--fp16", action="store_true", help="enable fp16")
    parser.add_argument("--limit-turns-to", default=None, type=int,
                        help="limits the number of turns in a context to the value when selecting num_eval_samples")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--datastore_for_simulation", type=str, default=None)
    parser.add_argument("--starting_index", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--percentdata_datastore", default=None, type=str)
    parser.add_argument("--percentdata_datafile", default=None, type=str)

    # FOP-only arguments
    parser.add_argument("--num_candidate_generations", default=None, type=int)
    parser.add_argument("--num_neighbors", default=1024, type=int)
    parser.add_argument("--num_futures", type=int, default=1)
    parser.add_argument("--to_compress", action="store_true",
                        help="Whether to use a compressed sentence when retrieving multiple futures. If flag not included, the first retrieved sentence is used.")

    # DBS-only arguments
    parser.add_argument("--number_of_beams", default=3, type=int)
    parser.add_argument("--number_of_concurrent_sentences", default=5, type=int)
    parser.add_argument("--number_of_generated_sentences", default=1, type=int)

    # FOP + DBS arguments
    parser.add_argument("--lambda_param", type=int, default=None)

    # Prompting with training arguments
    parser.add_argument("--prompting_model_path", type=str, default=None)

    args = parser.parse_args()
    if args.no_cuda:
        DEVICE = "cpu"

    if args.no_cuda:
        DEVICE = "cpu"

    fileConfig("../logger_config.conf")
    start = timer()
    run_approach(args)
    end = timer()
    logger.info(f"Time to run approaches: {(end - start)}")
