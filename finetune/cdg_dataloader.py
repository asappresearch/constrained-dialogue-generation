import abc
import logging
import os
import random
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange

from approaches.helpers import get_cust_utts_future, get_cust_utts_past, count_keywords, set_seed
from approaches.run_approaches import get_conv_ngrams, get_tfidf_vectorizer, get_tfidf_topn, \
    get_preprocessed_cust_utt
from approaches.run_approaches import load_data, load_datastore

from datastore.knn_datastore import preprocess_for_datastore, DataStore

from torch.utils.data import Dataset

set_seed(1234)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EOS = "<|endoftext|>"
IGNORE_INDEX = -100
N_TURNS_PER_CTX = 5

logger = logging.getLogger("application.CDGDataLoader")


class CDGDataLoader(Dataset):
    def __init__(self, config, model, tokenizer, split="train"):
        self.load_examples(config, model, tokenizer, split)

    def load_examples(self, config, model, tokenizer, split):
        assert "DATALOADER" in config
        dataloader_config = config["DATALOADER"]

        if split == "eval":
            assert config.get("DATALOADER", "eval_data_file", fallback=None) is not None
            assert os.path.isfile(dataloader_config["eval_data_file"])

            directory, filename = os.path.split(dataloader_config["eval_data_file"])
            dest_file = config['TRAINING']['eval_keywords_file_path']
        elif split == "train":
            assert config.get("DATALOADER", "train_data_file", fallback=None) is not None
            assert os.path.isfile(dataloader_config["train_data_file"])
            dest_file = config['TRAINING']['train_keywords_file_path']

            directory, filename = os.path.split(dataloader_config["train_data_file"])
        else:
            raise ValueError

        assert config.get("DATALOADER", "cache_file_postfix", fallback=None) is not None
        num_keywords = None
        if config.get("DATALOADER", "num_keywords", fallback=None) is not None:
            num_keywords = int(dataloader_config["num_keywords"])
        os.makedirs(config['TRAINING']['output_dir'], exist_ok=True)

        num_examples = int(dataloader_config["num_examples"])
        if num_examples > 0:
            cached_features_file = os.path.join(config['TRAINING']['output_dir'],
                                                "cached_" + dataloader_config['include_future'] + "_" +
                                                dataloader_config['future_type'] + "_" + split + "_" +
                                                dataloader_config["cache_file_postfix"] + "_examples" + str(
                                                    num_examples) + ".pkl")
        elif num_examples == -1:
            cached_features_file = os.path.join(config['TRAINING']['output_dir'],
                                                "cached_" + dataloader_config['include_future'] + "_" +
                                                dataloader_config['future_type'] + "_" + split + "_" +
                                                dataloader_config["cache_file_postfix"] + ".pkl")

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            self.examples = load_data(cached_features_file)
        else:
            logger.info("Processing preprocess_data")
            if config.get("DATALOADER", "prompt_type", fallback=None) is not None:
                keywords_dir = dataloader_config["keywords_dir"]
                prompt_type = dataloader_config["prompt_type"]
                if not os.path.exists(dest_file):
                    constructed_data = self.construct_data(config, split, model, tokenizer)
                    train_constructed_data = self.construct_data(config, "train", model,
                                                                 tokenizer) if split == "eval" else None
                    assert keywords_dir is not None
                    os.makedirs(keywords_dir, exist_ok=True)
                    self.find_keywords(constructed_data, config, split, keywords_dir, train_constructed_data, dest_file=dest_file)

            self.examples = []
            self.create_dataset(config, split, model, tokenizer, cached_features_file, num_examples)

    def construct_data(self, config, split, model, tokenizer):
        if split == "train":
            data = load_data(config['DATALOADER']["train_data_file"], split)
        elif split == "eval":
            data = load_data(config['DATALOADER']["eval_data_file"], split)
        
        constructed_data = {}
        for issueid in tqdm(data):
            text, _, _ = self.get_conv_and_mask(tokenizer, data[issueid], stop_index=len(data[issueid]['spkr']),
                                                block_size=1024)
            text_str = tokenizer.decode(text)
            constructed_data[issueid] = text_str
        return constructed_data

    def find_keywords(self, raw_data, config, split, keywords_dir, train_constructed_data, dest_file=None):
        prompt_type = config.get("DATALOADER", "prompt_type", fallback=None)
        if prompt_type is None:
            return

        dataloader_config = config["DATALOADER"]

        data = {issueid: get_preprocessed_cust_utt(raw_data[issueid]) for issueid in tqdm(raw_data)}
        if split == "train":
            train_data = data
        elif split == "eval":
            train_data = {issueid: get_preprocessed_cust_utt(train_constructed_data[issueid]) for issueid in
                          train_constructed_data}

        # Get keywords depending on prompt_type 
        if prompt_type == "conversation-tfidf":
            if dest_file is None:
                dest_file = os.path.join(keywords_dir, split + "_" + prompt_type + ".pkl")

            if os.path.isfile(dest_file):
                return

            corpus = [" ".join(train_data[issueid]) for issueid in train_data]
            tfidf_vectorizer = get_tfidf_vectorizer(corpus, min_df=1)

            keywords = {issueid: get_tfidf_topn(" ".join(data[issueid]), tfidf_vectorizer) for issueid in tqdm(data)}
            torch.save(keywords, dest_file)
        else:
            assert False

        return

    def get_conv_and_mask(self, tokenizer, example, stop_index, block_size, keywords_prompt=None, keyword_len=0):
        text = []
        mask_customer = []
        mask_agent = []

        if keywords_prompt is not None:
            keyword_text = tokenizer.encode("__KEYWORDS__") + tokenizer.encode(keywords_prompt)
            if len(keyword_text) < keyword_len:
                keyword_text += [tokenizer.pad_token_id] * (keyword_len-len(keyword_text))
            else:
                keyword_text = keyword_text[:keyword_len]

            text = keyword_text + text
            mask_customer = [False] * keyword_len + mask_customer
            mask_agent = [False] * keyword_len + mask_agent

        for i, (utt, spkr) in enumerate(zip(example["text"], example["spkr"])):
            if i > stop_index:
                break
            tmp = tokenizer.encode(f" __{spkr.upper()}__ {utt} {tokenizer.eos_token}", add_prefix_space=True)
            text += tmp
            if i < stop_index:
                mask_customer += [False] * len(tmp)
                mask_agent += [False] * len(tmp)
            elif i == stop_index:
                if spkr == "agent":
                    import pdb; pdb.set_trace()
                    logger.info("Should not be happening as we are training for customer only now.")
                    assert False
                else:
                    mask_customer += [False] + [True] * (len(tmp) - 1)
                    mask_agent += [False] * len(tmp)
        n_pad = block_size - len(text)

        if n_pad > 0:
            text += [tokenizer.pad_token_id] * n_pad
            mask_agent += [False]*n_pad
            mask_customer += [False]*n_pad
        elif n_pad < 0:
            n_remove = -1*n_pad
            text = text[n_remove:]
            mask_agent = mask_agent[n_remove:]
            mask_customer = mask_customer[n_remove:]
        return text, mask_customer, mask_agent

    def create_dataset(self, config, split, model, tokenizer, cache_file_name, num_examples):
        dataloader_config = config["DATALOADER"]
        assert config.has_option("DATALOADER", "block_size")
        block_size = int(dataloader_config["block_size"])
        prompting = dataloader_config.getboolean("prompting", fallback=False)

        future_selection_type = dataloader_config.get('future_type', fallback=None)

        if split == "train":
            if dataloader_config['train_data_file'].endswith(".pkl"):
                data = torch.load(dataloader_config['train_data_file'])
            elif dataloader_config['train_data_file'].endswith(".json"):
                raw_data = json.load(open(dataloader_config['train_data_file'], 'r'))['train']
                data = {}
                for d in raw_data:
                    dct = {'spkr': [], 'text': []}
                    for spkr, text in d['original']:
                        if spkr == "action":
                           continue
                        dct['spkr'].append(spkr)
                        dct['text'].append(text)
                    data[d['convo_id']] = dict(dct)
            else:
                logger.error(dataloader_config['train_data_file'])
                assert False

            f_context_path = os.path.join(dataloader_config['datastore_file'], "context_data_train.csv")
            if not os.path.isfile(f_context_path):
                os.makedirs(dataloader_config['datastore_file'], exist_ok=True)
                f_processed_ds_data = os.path.join(dataloader_config['datastore_file'], "preprocessed_train_data.txt")
                _, _ = preprocess_for_datastore(dataloader_config['train_data_file'],
                                                f_processed_ds_data,
                                                "train")
                _ = DataStore.prepare_contexts(f_processed_ds_data, dataloader_config['datastore_file'], "train")

            context_data = pd.read_csv(f_context_path)

            if prompting:
                guide_words = load_data(config['TRAINING']['train_keywords_file_path'])
        elif split == "eval":
            if dataloader_config['eval_data_file'].endswith(".pkl"):
                data = torch.load(dataloader_config['eval_data_file'])
            elif dataloader_config['eval_data_file'].endswith(".json"):
                raw_data = json.load(open(dataloader_config['eval_data_file'], 'r'))['test']
                data = {}
                for d in raw_data:
                    dct = {'spkr': [], 'text': []}
                    for spkr, text in d['original']:
                        if spkr == "action":
                           continue
                        dct['spkr'].append(spkr)
                        dct['text'].append(text)
                    data[d['convo_id']] = dict(dct)
            else:
                logger.error(dataloader_config['eval_data_file'])
                assert False

            f_context_path = os.path.join(dataloader_config['datastore_file'], "context_data_test.csv")
            if not os.path.isfile(f_context_path):
                os.makedirs(dataloader_config['datastore_file'], exist_ok=True)
                f_processed_ds_data = os.path.join(dataloader_config['datastore_file'], "preprocessed_test_data.txt")
                _, _ = preprocess_for_datastore(dataloader_config['eval_data_file'],
                                                f_processed_ds_data,
                                                "test")
                _ = DataStore.prepare_contexts(f_processed_ds_data, dataloader_config['datastore_file'], "test")
            context_data = pd.read_csv(f_context_path)

            if prompting:
                guide_words = load_data(config['TRAINING']['eval_keywords_file_path'])

        for row in context_data.itertuples():
            try:
                issueid, turn_num = int(row.issue_ids), int(row.current_turn_idx)
            except:
                import pdb;
                pdb.set_trace()

            keywords_prompt=None
            keywords_len=0
            if prompting:
                num_keywords = dataloader_config.getint("num_keywords", fallback=None)
                keywords_len = dataloader_config.getint("keyword_length", fallback=None)
                assert num_keywords is not None and keywords_len is not None

                if issueid not in guide_words:
                    continue
                keywords_prompt = guide_words[issueid]
                if type(keywords_prompt) == tuple and len(keywords_prompt) == 2:
                    keywords_prompt = keywords_prompt[1]
                if type(keywords_prompt) == list:
                    keywords_prompt = " ".join(keywords_prompt[:num_keywords])
                elif isinstance(keywords_prompt, str):
                    _keywords = keywords_prompt.split(",") if "," in keywords_prompt else keywords_prompt.split()
                    keywords_prompt = " ".join(_keywords[:num_keywords])

            tokenized_text, mask_customer, mask_agent = self.get_conv_and_mask(tokenizer,
                                                                               data[issueid],
                                                                               turn_num,
                                                                               block_size,
                                                                               keywords_prompt=keywords_prompt,
                                                                               keyword_len=keywords_len)

            if not torch.any(torch.tensor(mask_customer)):
                continue

            self.examples.append((issueid,
                                  torch.tensor(tokenized_text),
                                  torch.tensor(mask_customer),
                                  torch.tensor(mask_agent)))

            if num_examples != -1 and len(self.examples) >= num_examples:
                break

        logger.info("Saving features into cached file %s", cache_file_name)
        curr_len_examples = len(self.examples)
        logger.debug(f'TOTAL EXAMPLE SIZE {curr_len_examples}')
        torch.save(self.examples, cache_file_name)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
