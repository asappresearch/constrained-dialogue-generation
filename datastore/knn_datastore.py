import csv
import json
import logging
import os
import sys
import pickle
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from logging.config import fileConfig

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fire import Fire
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
sys.path.append("../")
from approaches.helpers import set_seed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EOS = "<|endoftext|>"
IGNORE_INDEX = -100
N_TURNS_PER_CTX = 1

logger = logging.getLogger("application.KNNDatastore")

@dataclass
class TrainDstoreArgs:
    n_contexts: int
    faiss_index_file: str # faiss index output file
    faiss_future_index_file: str # faiss future index output file
    starting_point: int # index to start adding keys at
    kv_file_prefix: str = None
    dimension: int = 1024
    seed: int = 1234
    ncentroids: int = 2048 # number of centroids faiss should learn
    qvector_sz: int = 64 # size of the quantized vectors
    n_probes: int = 8 # number of clusters to query
    n_keys_to_add_at_a_time: int = 1000000 # can only load a certain amount of preprocess_data to memory at a time
    n_bits: int = 8
    verbose: bool = True


class Printer:
    @staticmethod
    def print(str, verbose_flag=True):
        if verbose_flag:
            logger.info(str)


class GPT2Encoder(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2Encoder, self).__init__(config)

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        transformer_outputs = self.transformer(input_ids,
                                              past_key_values=past_key_values,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              position_ids=position_ids,
                                              head_mask=head_mask,
                                              inputs_embeds=inputs_embeds,
                                              encoder_hidden_states=encoder_hidden_states,
                                              encoder_attention_mask=encoder_attention_mask,
                                              use_cache=use_cache,
                                              output_attentions=output_attentions,
                                              output_hidden_states=output_hidden_states,
                                              return_dict=return_dict)
        hidden_states = transformer_outputs.last_hidden_state[:,-1,:]

        return hidden_states


def preprocess_conv(tokenizer, conv_batch):
    tokens = [tokenizer.encode(l) for l in conv_batch]

    max_len = 1024
    for i, t in enumerate(tokens):
        if len(t) > max_len:
            tokens[i] = t[-max_len:]

    prompt_batch = [tokenizer.decode(t) for t in tokens]
    try:
        seq_lens = max([len(t) for t in tokens])
    except:
        import pdb
        pdb.set_trace()

    encodings_dict = tokenizer.batch_encode_plus(prompt_batch, max_length=seq_lens, truncation=True,
                                                 padding='max_length', add_prefix_space=True)

    input_ids = torch.tensor(encodings_dict["input_ids"]).to(DEVICE)
    attn_mask = torch.tensor(encodings_dict["attention_mask"]).to(DEVICE)

    last_non_masked_idx = (torch.sum(attn_mask, dim=1) - 1).to(DEVICE)
    start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, len(tokenizer)).unsqueeze(1).to(DEVICE)
    position_ids = torch.tensor([list(range(seq_lens)) for i in range(input_ids.shape[0])]).to(DEVICE)

    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]].clone()

    return input_ids, attn_mask, position_ids


def process_traindata(data):
    all_conversations = [v for v in data.values()]
    all_keys = [k for k in data.keys()]
    processed_conversations = []
    n_turns = []
    for conversation in all_conversations:
        if isinstance(conversation, dict):
            iterator = zip(conversation['spkr'], conversation['text'])
        elif isinstance(conversation, list):
            assert len(conversation[0]) == 2
            iterator = conversation
        else:
            raise ValueError("Unknown instance type for conversation: {type(conversation)}")

        conv = []
        for spkr, text in iterator:
            if spkr == "action":
                continue

            conv.append(f"__{spkr.upper()}__ {text} {EOS}")

        n_turns.append(len(conv))
        processed_conversations.append(" ".join(conv))

    logger.info(f"min turns: {min(n_turns)}")
    logger.info(f"max turns: {max(n_turns)}")
    logger.info(f"avg turns: {np.mean(n_turns)}")

    return all_keys, processed_conversations

class DataStore:
    def __init__(self, args):
        self.fp16 = args.fp16
        self.dimension = args.embed_dim
        self.k = args.k
        self.n_contexts = args.n_contexts
        self.index = None
        self.keys = None
        self.vals = None

    def load_index(self, index_file, future_index_file=None, verbose=True):
        assert os.path.exists(index_file)

        start_time = time.time()
        self.index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        if future_index_file is not None:
            self.future_index = faiss.read_index(future_index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)

        Printer.print(f"loading the indexes took {time.time()-start_time} s", verbose)

    def load_kvstore_to_memory(self, file_prefix, verbose=True):
        assert self.n_contexts != 0
        assert self.dimension != 0

        assert os.path.exists(f"{file_prefix}_keys.npy")
        assert os.path.exists(f"{file_prefix}_vals.npy")

        start_time = time.time()

        try:
            keys_from_mmap = np.memmap(f"{file_prefix}_keys.npy", dtype=np.float16 if self.fp16 else np.float32, mode='r', shape=(self.n_contexts, self.dimension))
            vals_from_mmap = np.memmap(f"{file_prefix}_vals.npy", dtype=np.float16 if self.fp16 else np.float32, mode='r', shape=(self.n_contexts, self.dimension))
        except:
            import pdb; pdb.set_trace()

        self.keys = np.zeros((self.n_contexts, self.dimension), dtype=np.float16 if self.fp16 else np.float32)
        self.keys = keys_from_mmap[:].astype(np.float16 if self.fp16 else np.float32)

        self.vals = np.zeros((self.n_contexts, self.dimension), dtype=np.float16 if self.fp16 else np.float32)
        self.vals = vals_from_mmap[:].astype(np.float16 if self.fp16 else np.float32)

        Printer.print(f"Loading key value pairs to memory took: {time.time()-start_time} s", verbose)

    def load_kv_memmap(self, file_prefix):
        self.keys = np.memmap(f"{file_prefix}_keys.npy", dtype=np.float16 if self.fp16 else np.float32, mode='r', shape=(self.n_contexts, self.dimension))
        self.vals = np.memmap(f"{file_prefix}_vals.npy", dtype=np.float16 if self.fp16 else np.float32, mode='r', shape=(self.n_contexts, self.dimension))

    def get_knns(self, queries, index_type="past", k=None):
        index = self.future_index if index_type=="future" else self.index
        start = time.time()
        if torch.is_tensor(queries):
            queries = queries.detach().cpu().float().numpy()
        k = self.k if k is None else k
        dists, knns = index.search(queries.astype(np.float32), k)
        return dists, knns

    @staticmethod
    def prepare_contexts(f_data, out_path, split, total_line_count=None):
        """
        parameters:
            - f_data: path to preprocessed preprocess_data
            - out_path: directory path to store the context preprocess_data
            - split: [train, test]
            - total_line_count: total number of contexts. Only used for tqdm so it's optional.
        """
        num_ctxs = 0
        first_ctx = True
        with open(f_data, 'r') as f:
            if total_line_count is not None:
                iterator = tqdm(f, desc="Preparing Contexts", total=total_line_count)
            else:
                iterator = tqdm(f, desc="Preparing Contexts")

            for line in iterator:
                if line.strip() == "":
                    continue
                try:
                    _issueid, ctx = line.split(",", 1)
                except:
                    import pdb;
                    pdb.set_trace()

                turns = ctx.split(EOS)
                if len(turns) < N_TURNS_PER_CTX:
                    continue

                dct = {'contexts': [], 'issue_ids': [], 'current_turn_idx': [], 'future': []}
                for t in range(N_TURNS_PER_CTX, len(turns), 1):
                    if '__AGENT__' in turns[t]:
                        continue

                    ctx_str = ' '.join(turns[:t])
                    future_str = ' '.join(turns[t:])

                    dct['contexts'].append(ctx_str)
                    dct['issue_ids'].append(_issueid)
                    dct['current_turn_idx'].append(t)
                    dct['future'].append(future_str)

                    num_ctxs += 1

                with open(os.path.join(out_path, f'context_data_{split}.csv'), 'a') as f:
                    tmp_df = pd.DataFrame(dct)
                    # Add endoftext tokens and clean up additional whitespace to match with other code that uses similar contexts
                    tmp_df['contexts'] = tmp_df.apply(lambda row: add_endoftext_tokens(row['contexts']), axis=1)
                    tmp_df['future'] = tmp_df.apply(lambda row: add_endoftext_tokens(row['future']), axis=1)
                    tmp_df.to_csv(f, index=False, header=first_ctx)

                first_ctx = False

        return num_ctxs

    def get_nearest_neighbors(self, model, tokenizer, input_strs):
        input_ids, attn_mask, position_ids = preprocess_conv(tokenizer, input_strs)
        transformer_outputs = model.transformer(
            input_ids,
            past_key_values=None,
            attention_mask=attn_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True)
        hidden_states = transformer_outputs[0]
        lm_logits = model.lm_head(hidden_states)
        m = torch.nn.Softmax(dim=2)
        prob_lm = m(lm_logits)[0]
        logger.info(prob_lm)

        hidden_states = transformer_outputs.last_hidden_state[:,-1,:]
        hidden_states = hidden_states.view(input_ids.size(0), 1, -1)

        logger.debug(hidden_states.shape)
        target = input_ids[:,-1]
        target = target.view(target.size(0), 1)

        _, knns = self.get_knns(hidden_states)
        return knns[0]

    def build_index(self, args: TrainDstoreArgs):
        if self.keys is None:
            self.load_kv_memmap(args.kv_file_prefix)

        if not os.path.exists(args.faiss_index_file):
            quantizer = faiss.IndexFlatL2(args.dimension)
            index = faiss.IndexIVFPQ(quantizer, args.dimension, args.ncentroids, args.qvector_sz, args.n_bits)
            index.nprobe = args.n_probes

            Printer.print("Training Index", args.verbose)
            set_seed(args.seed)
            random_sample = np.random.choice(np.arange(self.vals.shape[0]), 
                                size=[min(1000000, self.vals.shape[0])], 
                                replace=False)
            start_time = time.time()
            # faiss doesn't handle fp16 well
            index.train(self.keys[random_sample].astype(np.float32))
            Printer.print(f'Training took {time.time()-start_time} s', args.verbose)
            
            Printer.print("Writing index after training", args.verbose)
            start_time = time.time()
            faiss.write_index(index, args.faiss_index_file)
            Printer.print(f"writing index took {time.time() - start_time} s", args.verbose)

        Printer.print("Adding keys", args.verbose)
        index = faiss.read_index(args.faiss_index_file)
        start = args.starting_point
        start_time = time.time()
        pbar = tqdm(total=args.n_contexts, desc="Completion")
        while start < args.n_contexts:
            end = min(args.n_contexts, start+args.n_keys_to_add_at_a_time)
            keys_to_add = self.keys[start:end].copy()
            index.add_with_ids(keys_to_add.astype(np.float32), np.arange(start, end))
            start += args.n_keys_to_add_at_a_time

            if start % 1000000 == 0:
                Printer.print(f'Added {start} tokens so far', args.verbose)
                Printer.print(f'Writing Index {start}', args.verbose)
                faiss.write_index(index, args.faiss_index_file)

            pbar.update(args.n_keys_to_add_at_a_time)

        Printer.print(f"Added {start} tokens so far")
        Printer.print(f'Adding took {time.time()-start_time} s')
        Printer.print('Writing final Index')
        start_time = time.time()
        faiss.write_index(index, args.faiss_index_file)
        Printer.print(f'Writing final index took {time.time()-start_time} s')

    def build_future_index(self, args: TrainDstoreArgs):
        if self.vals is None:
            self.load_kv_memmap(args.kv_file_prefix)

        if not os.path.exists(args.faiss_future_index_file):
            quantizer = faiss.IndexFlatL2(args.dimension)
            future_index = faiss.IndexIVFPQ(quantizer, args.dimension, args.ncentroids, args.qvector_sz, args.n_bits)
            future_index.nprobe = args.n_probes

            Printer.print("Training Index", args.verbose)
            set_seed(args.seed)
            random_sample = np.random.choice(np.arange(self.vals.shape[0]), 
                                size=[min(1000000, self.vals.shape[0])], 
                                replace=False)
            start_time = time.time()
            # faiss doesn't handle fp16 well
            future_index.train(self.vals[random_sample].astype(np.float32))
            Printer.print(f'Training took {time.time()-start_time} s', args.verbose)
            
            Printer.print("Writing index after training", args.verbose)
            start_time = time.time()
            faiss.write_index(future_index, args.faiss_future_index_file)
            Printer.print(f"writing index took {time.time() - start_time} s", args.verbose)

        Printer.print("Adding keys", args.verbose)
        future_index = faiss.read_index(args.faiss_future_index_file)
        start = args.starting_point
        start_time = time.time()
        pbar = tqdm(total=args.n_contexts, desc="Completion")
        while start < args.n_contexts:
            end = min(args.n_contexts, start+args.n_keys_to_add_at_a_time)
            keys_to_add = self.vals[start:end].copy()
            future_index.add_with_ids(keys_to_add.astype(np.float32), np.arange(start, end))
            start += args.n_keys_to_add_at_a_time

            if start % 1000000 == 0:
                Printer.print(f'Added {start} tokens so far', args.verbose)
                Printer.print(f'Writing Index {start}', args.verbose)
                faiss.write_index(future_index, args.faiss_future_index_file)

            pbar.update(args.n_keys_to_add_at_a_time)

        Printer.print(f"Added {start} tokens so far")
        Printer.print(f'Adding took {time.time()-start_time} s')
        Printer.print('Writing final Index')
        start_time = time.time()
        faiss.write_index(future_index, args.faiss_future_index_file)
        Printer.print(f'Writing final index took {time.time()-start_time} s')

    @staticmethod
    def build_kvstore(encoder_type, model_path, f_data, batch_size, file_prefix, fp16,
                      dimension, out_path, split, total_line_count=None, context_prep=True, parallelize=False,
                      finetuned=False, additional_tokens_path=None):
        logger = logging.getLogger("application.KNNDatastore.build_kvstore")
        
        key_dtype = np.float16 if fp16 else np.float32
        future_dtype = np.float16 if fp16 else np.float32

        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = eval(encoder_type).from_pretrained(model_path).to(DEVICE)

        if not finetuned:
            assert additional_tokens_path is not None

            additional_tokens = []
            with open(additional_tokens_path, 'r') as f:
                for line in f.readlines():
                    for word in line.split(","):
                        word = word.strip()
                        additional_tokens.append(word)

            tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens,
                                          "pad_token": "__PAD__"})
            model.resize_token_embeddings(len(tokenizer))

        if torch.cuda.device_count() > 1 and parallelize:
            logger.info(f"There are {torch.cuda.device_count()} GPUs available")
            model = nn.DataParallel(model)
        elif  fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model = amp.initialize(model, opt_level="O2")

        input_ids = None
        past_key_values = None
        attn_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        labels = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None

        if context_prep:
            num_ctxs = DataStore.prepare_contexts(f_data, out_path, split, total_line_count)
        else:
            logger.info("Calculating the number of contexts")
            num_ctxs = 0
            with open(os.path.join(out_path, f"context_data_{split}.csv"), 'r') as f:
                for _ in csv.reader(f):
                    num_ctxs += 1

            num_ctxs = num_ctxs - 1  # remove the header line

        n_contexts = num_ctxs

        logger.info(f"Total number of contexts to be stored: {num_ctxs}")

        keys = np.memmap(f"{out_path}/{file_prefix}_{split}_keys.npy", dtype=key_dtype, mode='w+', shape=(n_contexts, dimension))
        vals = np.memmap(f"{out_path}/{file_prefix}_{split}_vals.npy", dtype=future_dtype, mode='w+', shape=(n_contexts, (dimension)))
        arr_issueid_currentidx = np.zeros((n_contexts, 2), dtype=np.long)

        dstore_idx = 0
        total_runs = n_contexts // batch_size
        pbar = tqdm(total=total_runs+1, desc="Processing Contexts")
        batch_id = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(os.path.join(out_path, f"context_data_{split}.csv"), chunksize=batch_size)):
            with torch.no_grad():
                input_ids, attn_mask, position_ids = preprocess_conv(
                    tokenizer=tokenizer,
                    conv_batch=chunk['contexts']
                )

                hidden_states = model(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                ).detach().cpu()

                hidden_states = hidden_states.view(input_ids.size(0), -1)

                assert hidden_states.shape == (input_ids.size(0), dimension)

                end_idx = batch_id + len(chunk['contexts'])

                keys[batch_id:end_idx] = hidden_states.cpu().numpy().astype(key_dtype)

                del input_ids
                del attn_mask
                del position_ids
                del hidden_states

                torch.cuda.empty_cache()

                future_input_ids, future_attn_mask, future_position_ids = preprocess_conv(
                    tokenizer=tokenizer,
                    conv_batch=chunk['future']
                )

                future_hidden_states = model(
                    future_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=future_attn_mask,
                    token_type_ids=token_type_ids,
                    position_ids=future_position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                ).detach().cpu()

                future_hidden_states = future_hidden_states.view(future_input_ids.size(0), -1)

                assert future_hidden_states.shape == (future_input_ids.size(0), dimension)

                for i, issueid, current_turn_idx, encoded_future in zip(range(batch_id, end_idx), chunk['issue_ids'], chunk['current_turn_idx'], future_hidden_states):
                    array = np.zeros((dimension), dtype=future_dtype)
                    if isinstance(issueid, str):
                        arr_issueid_currentidx[i][0] = int(issueid)
                    else:
                        arr_issueid_currentidx[i][0] = issueid
                    arr_issueid_currentidx[i][1] = current_turn_idx

                    array = encoded_future.cpu().numpy().astype(future_dtype)
                    vals[i] = np.copy(array)
                    del array

                dstore_idx = end_idx

                del future_input_ids
                del future_attn_mask
                del future_position_ids
                del future_hidden_states

                torch.cuda.empty_cache()

            batch_id += batch_size
            if (chunk_idx+1)*batch_size != batch_id:
                import pdb; pdb.set_trace()

            pbar.update(1)
        pbar.close()

        torch.save(arr_issueid_currentidx, os.path.join(out_path, f"{file_prefix}_{split}_issueid_currentidx.pkl"))

        logger.info(f"dstore_idx: {dstore_idx}")
        logger.info(f"Keys: {keys.shape}, {keys.dtype}")
        logger.info(f"Vals: {vals.shape}, {vals.dtype}")

        assert keys.shape[0] == vals.shape[0]

        return keys.shape[0]


def add_endoftext_tokens(s, endoftext_token="<|endoftext|>"):
    new_s = ""
    for word in s.split():
        if (word == "__CUSTOMER__" or word == "__AGENT__") and len(new_s)>0:
            new_s += endoftext_token+" "
        new_s += word+" "
    new_s += endoftext_token
    return new_s.strip()

def preprocess_for_datastore(data_path, f_processed_data, split):
    """
    parameters:
        - data_path: path to the original preprocess_data .pkl or .json
        - f_processed_data: output filename for the processed preprocess_data
        - split: [train, test]
    """
    line_count = 0
    if data_path.endswith(".pkl"):
        # expects the format {issue_id: {spkr:[], text:[]}}
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        except:
            data = torch.load(data_path)
    elif data_path.endswith(".json"):
        assert split is not None
        # expects the format {split: {spkr: [], text:[]}}``
        with open(data_path, 'r') as f:
            raw_data = json.load(f)[split]
        data = {}
        for obj in raw_data:
            data[obj['convo_id']] = obj['original']
    else:
        raise ValueError("Unknown file extension in data_path")
    ids, processed_convs = process_traindata(data)
    re_spaces_pattern = re.compile("\s+")
    with open(f_processed_data, 'w') as f:
        for idx, line in zip(ids, processed_convs):
            line = line.strip("\n")
            line = re_spaces_pattern.sub(" ", line)
            f.write(f"{idx},{line}\n")
            line_count += 1

    return line_count, processed_convs

def main(build_datastore: bool, model_path: str, data_path: str, output_dir: str,
         split: str, encoder_type: str = "GPT2Encoder", create_index_only: bool = False,
         batch_size: int = 32, file_prefix: str = "kvstore",
         fp16: bool = False, embed_dim: int = 1024, k_neighbors: int = 1024,
         parallelize: bool = True, n_turns_per_context=1, num_contexts: int = None,
         seed: int = 1234, ncentroids: int = 4096, qvector_sz: int = 64, n_probes: int = 8,
         n_keys_to_add_at_a_time: int = 1000000, n_bits: int = 8, finetuned=False, additional_tokens_path=None
         ):

    set_seed(seed)
    if os.path.isfile("../logger_config.conf"):
        fileConfig("../logger_config.conf")
    
    logger = logging.getLogger("application.KNNDatastore")

    global N_TURNS_PER_CTX
    N_TURNS_PER_CTX = n_turns_per_context

    # Add every new encoder to this list to do a sanity check
    available_encoders = ["GPT2Encoder"]
    logger.info(f"ENCODER TYPE: {encoder_type}")
    assert encoder_type in available_encoders

    os.makedirs(output_dir, exist_ok=True)

    if create_index_only:
        assert num_contexts is not None
        modified_f_prefix = f"{file_prefix}_{split}"
        datastore_args = TrainDstoreArgs(
            n_contexts=num_contexts,
            faiss_index_file=os.path.join(output_dir, f'faiss_{split}_index.ind'),
            faiss_future_index_file=os.path.join(output_dir, f'faiss_{split}_future_index.ind'),
            starting_point=0,
            kv_file_prefix=f"{os.path.join(output_dir, modified_f_prefix)}",
            dimension=embed_dim,
            seed=seed, 
            ncentroids=ncentroids,
            qvector_sz=qvector_sz,
            n_probes=n_probes,
            n_keys_to_add_at_a_time=n_keys_to_add_at_a_time,
            n_bits=n_bits,
        )

        dargs = namedtuple('dargs', 'fp16 embed_dim k n_contexts')
        dstore = DataStore(dargs(
            fp16,
            embed_dim,
            k_neighbors,
            num_contexts
        ))
        logger.info("Building the faiss index")
        dstore.build_index(datastore_args)
        dstore.build_future_index(datastore_args)
    elif build_datastore:
        f_processed_data = os.path.join(output_dir, f"processed_data_{split}.txt")
        line_count = 0
        if os.path.isfile(f_processed_data):
            with open(f_processed_data, 'r') as f:
                for _ in f:
                    line_count += 1
        else:
            line_count, _ = preprocess_for_datastore(data_path, f_processed_data, split)

        logger.info("Building key-value store")
        context_prep_flag = not os.path.isfile(os.path.join(output_dir, f"context_data_{split}.csv"))
        n_contexts = DataStore.build_kvstore(encoder_type, model_path, f_processed_data, batch_size, file_prefix, fp16,
                                             embed_dim, output_dir, split, total_line_count=line_count,
                                             context_prep=context_prep_flag, parallelize=parallelize,
                                             finetuned=finetuned, additional_tokens_path=additional_tokens_path)
        logger.info("Building key-value store is complete")

        modified_f_prefix = f"{file_prefix}_{split}"
        datastore_args = TrainDstoreArgs(
            n_contexts=n_contexts,
            faiss_index_file=os.path.join(output_dir, f'faiss_{split}_index.ind'),
            faiss_future_index_file=os.path.join(output_dir, f'faiss_{split}_future_index.ind'),
            starting_point=0,
            kv_file_prefix=f"{os.path.join(output_dir, modified_f_prefix)}",
            dimension=embed_dim,
            seed=seed, 
            ncentroids=ncentroids,
            qvector_sz=qvector_sz,
            n_probes=n_probes,
            n_keys_to_add_at_a_time=n_keys_to_add_at_a_time,
            n_bits=n_bits,
        )

        dargs = namedtuple('dargs', 'fp16 embed_dim k n_contexts')
        dstore = DataStore(dargs(
            fp16,
            embed_dim,
            k_neighbors,
            n_contexts
        ))
        logger.info("Building the faiss index")
        dstore.build_index(datastore_args)
        dstore.build_future_index(datastore_args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    Fire(main)

