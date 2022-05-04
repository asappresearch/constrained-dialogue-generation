import json
import logging
import os
from collections import namedtuple
from configparser import ConfigParser
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import AdamW

from datastore.knn_datastore import TrainDstoreArgs, DataStore
from approaches.helpers import get_config_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger("application.Loader")


def load_models_and_tokenizers(config, task, checkpoint_path=None, model_class=None):
    training_type = config['TRAINING']['training_type']
    model_type = config['TRAINING']['model_name']
    if model_type == "GPT2":
        tokenizer_class = "GPT2Tokenizer"
        if model_class is None:
            if training_type == "fop":
                model_class = "FoP_GPT2LMHeadModel"
            else:
                model_class = "GPT2LMHeadModel"
    elif model_type == "pegasus":
        tokenizer_class = "PegasusTokenizer"
        if model_class is None:
            if training_type == "fop":
                model_class = "FoP_PegasusForConditionalGeneration"
            else:
                model_class = "PegasusForConditionalGeneration"
    else:
        raise NotImplementedError

    if checkpoint_path is not None:
        tokenizer = eval(tokenizer_class).from_pretrained(checkpoint_path)
        model = eval(model_class).from_pretrained(checkpoint_path)
    else:
        if task == "train":
            model_path = config['TRAINING']['model_type']
        elif task == "eval":
            if config.get("TRAINING", "eval_model_path", fallback=None) is not None:
                model_path = config["TRAINING"]["eval_model_path"]
            else:
                assert config.get("TRAINING", "output_dir", fallback=None) is not None
                model_path = config["TRAINING"]["output_dir"]
        else:
            raise ValueError

        model = eval(model_class).from_pretrained(model_path)
        tokenizer = eval(tokenizer_class).from_pretrained(model_path)

    block_size = int(config['DATALOADER']['block_size'])
    block_method = None
    if config.get('DATALOADER', 'block_method', fallback=None) is not None:
        block_method = config['DATALOADER']['block_method']

    if block_size <= 0:
        block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model

    additional_tokens = []
    if config.get('TRAINING', 'additional_tokens_path', fallback=None) is not None:
        with open(config['TRAINING']['additional_tokens_path'], 'r') as f:
            for line in f.readlines():
                for word in line.split(","):
                    word = word.strip()
                    additional_tokens.append(word)

    prompt_type = None
    if config.get("DATALOADER", "prompt_type", fallback=None) is not None:
        prompt_type = config["DATALOADER"]["prompt_type"]

    if prompt_type == "intent-id":
        issueid_to_intent = torch.load(config['DATALOADER']['issueid_to_intent_file'])
        all_intents = list(set(issueid_to_intent.values()))
        for intent in all_intents:
            if str(intent) != "nan":
                additional_tokens.append("__"+intent+"__")

    tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens, "pad_token": "__PAD__"})
    model.resize_token_embeddings(len(tokenizer))
    if block_method == "block-onehot":
        model.set_tag_embedding(2)

    return model, tokenizer


def load_config(config_path, kwargs=None):
    template = get_config_template()

    config = ConfigParser()
    config.read_dict(template)

    defaults_path = "experiment_configs/defaults.ini"
    if os.path.isfile(defaults_path):
        config.read(defaults_path)

    if config_path is not None:
        config.read(config_path)
    elif kwargs is not None:
        for key, val in kwargs.items():
            key = key.strip().lower()
            overridden = False
            for k, v in config.items():
                if key in v.keys():
                    if isinstance(val, bool):
                        if val:
                            config[k][key] = "yes"
                        else:
                            config[k][key] = "no"
                    else:
                        if not isinstance(val, str):
                            config[k][key] = str(val)
                        else:
                            config[k][key] = val
                    overridden = True
                    break
            
            if not overridden:
                raise ValueError(f'Unrecognized argument {key} provided')
    else:
        raise ValueError("Neither config path or arguments provided")

    for key in list(config.keys()):
        for k, v in list(config[key].items()):
            if v == '':
                config.remove_option(key, k)

    datastore_file = config.get("DATALOADER", "datastore_file", fallback=None)

    return config


def load_data(data_path, split=None):
    if data_path.endswith(".pkl"):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except:
            data = torch.load(data_path)
    elif data_path.endswith(".json"):
        assert split is not None
        # expects the format {split: {spkr: [], text:[]}}``
        with open(data_path, 'r') as f:
            if split == "eval":
                split = "test"
            logger.info(split)
            raw_data = json.load(f)[split]
        data = {}
        for obj in raw_data:
            data[obj['convo_id']] = {
                'text': [text for spkr,text in obj['original'] if spkr in ['agent', 'customer']],
                'spkr': [spkr for spkr,text in obj['original'] if spkr in ['agent', 'customer']],
            }
            if 'scenario' in obj:
                data[obj['convo_id']]['extra-info'] = (obj['scenario']['flow'], obj['scenario']['subflow'])

    return data


def load_dataloader_and_optimizers(model, train_dataset, train_config, local_rank, n_gpu):
    train_batch_size = int(train_config["per_gpu_train_batch_size"]) * max(1, n_gpu)
    
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    learning_rate = float(train_config["learning_rate"])
    adam_epsilon = float(train_config["adam_epsilon"])
    weight_decay = float(train_config["weight_decay"])

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    return train_dataloader, train_sampler, optimizer


def load_datastore(knnlm_path, k_neighbors, split, n_contexts, fp16=False, embed_dim=1024, faiss_index_type="past"): #faiss_index_type can be "past" or "future"
    split = "test" if split == "eval" else split
    logger.info(f"SPLIT: {split}")
    logger.info(f"LOAD DATASTORE N_CONTEXTS: {n_contexts}")
    future_index_file = os.path.join(knnlm_path, 'faiss_'+split+'_future_index.ind')
    if not os.path.exists(future_index_file):
        future_index_file = None
    datastore_args = TrainDstoreArgs(
        n_contexts=n_contexts,
        faiss_index_file=os.path.join(knnlm_path, 'faiss_'+split+'_index.ind'),
        faiss_future_index_file=future_index_file,
        starting_point=0,
        kv_file_prefix=f"{knnlm_path}/kvstore_"+split)
    dargs = namedtuple('dargs', 'fp16 embed_dim k n_contexts')
    dstore = DataStore(dargs(fp16, embed_dim, int(k_neighbors), n_contexts))
    logger.info(f"Float16 is being used: {dstore.fp16}")

    if split == "train":
        if faiss_index_type == "past":
            dstore.load_index(os.path.join(knnlm_path, f'faiss_{split}_index.ind'), future_index_file=future_index_file)
        elif faiss_index_type == "future":
            dstore.load_index(os.path.join(knnlm_path, f'faiss_{split}_future_index.ind'), future_index_file=future_index_file)

    dstore.load_kvstore_to_memory(f"{knnlm_path}/kvstore_{split}")
    logger.info(f"{dstore.keys.shape}")
    return dstore
