import torch
from tqdm import trange
import logging
import argparse
import os
import pandas as pd
import numpy as np
import random
from approaches.helpers import set_seed
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
set_seed(1234)

NOCUDA = os.getenv('NOCUDA', False)
DEVICE = "cuda" if torch.cuda.is_available() and not NOCUDA else "cpu"

def generate_all(conv_context, models, tokenizers, num_timesteps=10, num_convs=20,
                 num_tokens_to_produce=50, p=0.5, ngram_block=4, first_speaker="customer"):
    batch_size = min(num_convs, 10)
    curr_speaker = first_speaker  # Start with customer, unless otherwise specified
    prompt_text = [conv_context for i in range(num_convs)]
    only_generated_text = ["__" + curr_speaker.upper() + "__ " for i in range(num_convs)]
    for i in range(num_timesteps):
        generated_text = []
        for batch_start in range(0, len(prompt_text), batch_size):
            generated_text.extend(
                generate_response(prompt_text[batch_start: batch_start + batch_size], models, tokenizers,
                                  curr_speaker, num_tokens_to_produce, p, ngram_block)
            )
        # Switch speaker
        curr_speaker = "agent" if curr_speaker == "customer" else "customer"
        # Add generated text to prompt and repeat
        prompt_text = [prompt_text[k] + generated_text[k] + " <|endoftext|> __" + curr_speaker.upper() + "__ " for k
                       in range(len(prompt_text))]
        only_generated_text = [
            only_generated_text[k] + generated_text[k] + " <|endoftext|> __" + curr_speaker.upper() + "__ " for k in
            range(len(only_generated_text))]

    return only_generated_text


def generate_batch(prompt_batch, models, tokenizers, curr_speaker, num_tokens_to_produce, p, ngram_block):
    #    encode plus batch handles multiple batches and automatically creates attention_masks
    tokens = [tokenizers[curr_speaker].encode(l) for l in prompt_batch]
    max_len = 1024 - num_tokens_to_produce
    for i, t in enumerate(tokens):
        if len(t) > max_len:
            tokens[i] = t[-max_len:]
    prompt_batch = [tokenizers[curr_speaker].decode(t) for t in tokens]
    seq_lens = [len(t) for t in tokens]
    seq_len = max(seq_lens)
    assert seq_len <= max_len

    encodings_dict = tokenizers[curr_speaker].batch_encode_plus(
        prompt_batch, max_length=seq_len, truncation=True, padding='max_length', add_prefix_space=True
    )

    # ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
    input_ids = torch.tensor(encodings_dict["input_ids"]).to(models[curr_speaker].device)
    attn_mask = torch.tensor(encodings_dict["attention_mask"]).to(models[curr_speaker].device)

    generated_output = []
    if len(input_ids) > 1024:
        input_ids = input_ids[-1024:]

    output_sequences = models[curr_speaker].generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        min_length=input_ids.size(1) + 2,
        max_length=num_tokens_to_produce + input_ids.size(1),
        top_p=p,
        do_sample=True,
        no_repeat_ngram_size=ngram_block,
        pad_token_id=tokenizers[curr_speaker].pad_token_id,
        bad_words_ids=[
            tokenizers[curr_speaker].encode("__PAD__"),
            tokenizers[curr_speaker].encode("__AGENT__"),
            tokenizers[curr_speaker].encode("__CUSTOMER__"),
            tokenizers[curr_speaker].encode("__END_OF_TURN__"),
        ],
    )

    return output_sequences[:, input_ids.size(1):]


def generate_response(conv_context, models, tokenizers, curr_speaker,
                      num_tokens_to_produce, p, ngram_block):
    generated_outputs = []
    generated_output = generate_batch(conv_context, models, tokenizers, curr_speaker,
                                      num_tokens_to_produce, p, ngram_block)
    for i in range(generated_output.size(0)):
        generated_outputs.append(generated_output[i])
    generated_text = []
    for i in range(len(generated_outputs)):
        text = tokenizers[curr_speaker].decode(generated_outputs[i], skip_special_tokens=False)
        if '<|endoftext|>' in text:
            eos_index = text.index('<|endoftext|>')
            # Take the generated text up until the end of text token if it exists
            generated_text.append(text[:eos_index])
        else:
            generated_text.append(text)

    return generated_text
