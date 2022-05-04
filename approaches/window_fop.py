import logging
import math
import os
import re
import time
import numpy as np
import torch
from tqdm import trange

os.environ[
    'TRANSFORMERS_CACHE'] = './transformer_models'  # Work-around to avoid memory problems in server, comment out depending on memory availability

import torch.nn.functional as F
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from approaches.helpers import set_seed

porter = PorterStemmer()
IGNORE_INDEX = -100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("application.WindowFOP")

class GloveEncoder:
    encoder = api.load("glove-wiki-gigaword-300")

    @staticmethod
    def get_embedding(word):
        assert GloveEncoder.encoder is not None
        if word in GloveEncoder.encoder:
            # embedding = encoder[word]
            embedding = GloveEncoder.encoder[word]
        else:
            embedding = np.random.uniform(low=-2, high=2, size=(300,))

        return embedding

def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("'d", '')
    string = string.replace("?", '')
    string = string.replace("'s", '')
    string = string.replace(":", '')
    string = string.replace("!", '')
    string = string.replace('"', '')
    string = string.replace(".", '')
    string = string.replace("--", '')
    string = string.replace("'", '')
    string = string.replace(",", '')
    string = string.replace(';', '')
    string = string.replace('‘', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '')
    return (string)


## Pytorch
def converter_embedding_table(embedding_type, model, tokenizer, save_dir):
    save_path = os.path.join(save_dir, f"converter_table_{embedding_type}.npy")
    emb_dim = None
    if os.path.exists(save_path):
        return np.load(save_path)
    if embedding_type == "glove":
        encoder = api.load("glove-wiki-gigaword-300")
        emb_dim = 300
    elif embedding_type == "lm":
        encoder = model.transformer.wte.weight
        emb_dim = 1024

    # load gpt-2 model
    vocab_size = len(tokenizer)
    holder = np.zeros((vocab_size, emb_dim))
    # translate every word from the gpt-2 space into a glove representation
    for i in range(vocab_size):
        word = i
        if embedding_type == "glove":
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
        try:
            embedding = encoder[word]
            if embedding_type == "lm":
                embedding = embedding.cpu().detach().numpy()
            holder[i, :] = embedding
        except:
            holder[i, :] = np.random.uniform(low=-2, high=2, size=(emb_dim,))

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=save_path, arr=holder)
    logger.info('Table was generated')
    return holder


def count_word_stem(word, sequence):
    cust_utts = " ".join([x.split("<|endoftext|>")[0] for x in sequence.split("__CUSTOMER__")])
    sequence = cust_utts.split()
    word_count = 0

    word_stem = re.sub('[^A-Za-z]+', ' ', porter.stem(word).lower().strip()).strip()

    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        if (re.sub('[^A-Za-z]+', ' ', s_word_stem.lower().strip()).strip()  == word_stem):
            word_count += 1

    return word_count


# A score function for the quality of the sentence
def evaluate_quality(sequence, word, related_count, perplexity, guide, temp):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.001
    c_star = 2

    if (word == ""):
        quality_score = math.exp(-(w_1 * (c_star) + w_3 * perplexity))
        return quality_score

    quality_score = 0
    word_count = count_word_stem(word, sequence)

    if (word_count != 0) and guide:
        quality_score = math.exp(-(w_1 * word_count + w_3 * perplexity))
    else:
        quality_score = math.exp(-(w_1 * (c_star) + w_3 * perplexity))

    quality_score = quality_score / temp

    return quality_score, word_count


### End of the utils_gpt file
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def sample_sentence(text, this_sequence, orig_control_words, tokenizer, model, mask, keywords, encoded_keywords, curr_start, curr_end, converter_table, weight,
                    guide=False, prev_proba=1, top_k=0, top_p=0.9, temperature=1., only_max=False, upweight_control_words=False):
    """ Samples the next word of the sequence with logit modification (guidance)
    """
    indexed_tokens = tokenizer.encode(text)
    max_len = 1024  # - 50
    if len(indexed_tokens) > max_len:
        model_indexed_tokens = indexed_tokens[-max_len:]
    else:
        model_indexed_tokens = indexed_tokens
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([model_indexed_tokens])
    tokens_tensor = tokens_tensor.to(DEVICE)

    try:
        with torch.no_grad():
            outputs = model(tokens_tensor)
    except:
        import pdb;
        pdb.set_trace()
    del tokens_tensor

    torch.cuda.empty_cache()
    stemmed_control_words = [re.sub('[^A-Za-z]+', ' ', porter.stem(word).lower().strip()).strip() for word in orig_control_words]

    logits = outputs.logits
    logits = logits[0, -1, :] / temperature
    proba = F.softmax(logits, dim=-1)

    # Calculate cosine similarity
    if guide == True and curr_start >= 0 and curr_end <= len(keywords):
        count = 0
        for i in range(curr_start, curr_end, 1):
            if curr_start >= len(keywords) or curr_end > len(keywords):
                break
            assert len(keywords) == len(encoded_keywords)
            glove_word = encoded_keywords[i]
            word_order_multiplier = 1/(2**count)
            sim = cosine_similarity(np.reshape(glove_word, (1, -1)), converter_table)

            if only_max == True:
                sim_aux = np.zeros_like(sim)
                sim_aux[0, sim.argmax()] = sim.max()
                sim = sim_aux
            else:
                sim = np.clip(np.squeeze(sim), a_min=0, a_max=None)
            sim = sim * sim
            weighted_sim = torch.tensor(sim * weight * word_order_multiplier).to(DEVICE)
            stemmed_keyword = re.sub('[^A-Za-z]+', ' ', porter.stem(keywords[i]).lower().strip()).strip()
            if upweight_control_words and stemmed_keyword in stemmed_control_words:
                control_weight = np.zeros(sim.shape)
                control_tokens = tokenizer.encode(keywords[i])
                extra_control_tokens = tokenizer.encode(" "+keywords[i]) 
                if len(extra_control_tokens) == 1:
                    control_tokens = control_tokens + extra_control_tokens

                control_weight[control_tokens] = weight
                weighted_sim = weighted_sim + torch.tensor(control_weight).to(DEVICE)

            logits = logits + weighted_sim
            count += 1
            del weighted_sim
            del sim

    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()

    predicted_word = tokenizer.decode([predicted_index])
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]

    pred_word_stem = porter.stem(pred_word)

    del logits
    del proba
    torch.cuda.empty_cache()
    return predicted_text, this_sequence, predicted_word, pred_word_stem


def encode_keywords(embedding_type, model, tokenizer, word_index, index_keywords, save_dir):
    save_path = os.path.join(save_dir, f"{embedding_type}_set_{word_index}.npy")

    logger.info(f"Generating {embedding_type} embeddings for keywords...")
    if embedding_type == "glove":
        pass
    elif embedding_type == "lm":
        encoder = model.transformer.wte.weight

    encoded_words = []
    for word in index_keywords:
        if embedding_type == "glove":
            embedding = GloveEncoder.get_embedding(word)
        elif embedding_type == "lm":
            # this is slow since we commented the caching of embedding results
            assert False
            token = tokenizer.encode([word])
            if len(token) > 1:
                import pdb;
                pdb.set_trace()
            embedding = encoder[token].cpu().detach().numpy()
        encoded_words.append(embedding)
    np.save(save_path, encoded_words)
    return encoded_words


def conditional_language_generation(
        model,
        tokenizer,
        seed=None,
        nsamples=1,
        batch_size=1,
        length=None,
        temperature=1,
        top_k=0,
        top_p=0.9,
        models_dir='models',
        constant=20,
        number_of_concurrent_sentences=10,
        number_of_generated_sentences=20,
        number_of_words_per_sentence=5,
        number_of_beams=3,
        word_index=None,
        index_keywords=[],
        save_path='preprocess_data/dummy',
        sample=False,
        temp=1.,
        only_max=False,
        key2article=False,
        ROC=False,
        final_text_file="preprocess_data/final_text.txt",
        num_keywords=None,
        embedding_type="glove",
        agent_model_path="",
        first_utt="",
        window_size=4,
        orig_control_words=None,
        upweight_control_words=False,
        debug=False
):
    """
    Main function for conditional language generation
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    :top_p=1 Top_p is the cummulative probability used for nucleus sampling. 1 means no nucleus sampling
    :constant: How much are anchors weighted
    :counter index of wordset which is currently evaluated
    :TODO ns.....
    """
    start_time = time.time()

    total_words = number_of_words_per_sentence * number_of_generated_sentences

    if number_of_generated_sentences > 1:
        if agent_model_path == "":
            raise ValueError("agent_model_path should be provided for DBS")

    # Get encoded vocabulary
    converter_table = converter_embedding_table(embedding_type, model, tokenizer, save_path)
    # Get encoded keywords

    if len(index_keywords)>0:
        encoded_keywords = encode_keywords(embedding_type, model, tokenizer, word_index, index_keywords[0:num_keywords], save_dir=save_path)
    else:
        encoded_keywords = None
    keywords = index_keywords[0:num_keywords]

    try:
        if encoded_keywords is not None:
            assert len(index_keywords) == num_keywords
            assert len(encoded_keywords) == num_keywords
            assert len(index_keywords) == len(encoded_keywords)
    except:
        import pdb; pdb.set_trace()
    if encoded_keywords is not None:
        logger.info(f"Keywords: {keywords}, {len(encoded_keywords)}, {len(keywords)}")

    # Change full_text to desired initial context
    full_text = [f"{first_utt} "] * number_of_beams  # "I think"
    full_mask = [[False]] * number_of_beams

    number_keywords = len(keywords)

    from datetime import datetime
    # To measure computation time
    now = datetime.now()

    # prepare variables...
    set_seed(seed)
    weight = constant

    guidance_index = [0] * number_of_beams
    cum_quality_score = [0] * number_of_beams
    success_length = [0] * number_of_beams
    online_probability = [1] * number_of_beams
    guide = [True] * number_of_beams
    guidance_word = None
    
    for k in range(number_of_generated_sentences):
        # Define guidance word and index for guidance in model and quality function
        result_subsequences = []
        for b in range(number_of_beams):
            # Reset variables:
            beam_text = full_text[b]
            beam_mask = full_mask[b]

            perplexities = np.zeros((number_of_concurrent_sentences))

            ####################################### Generation loop ################################################
            for i in range(number_of_concurrent_sentences):
                context = beam_text
                context_mask = list(beam_mask)
                proba = 1
                this_sequence = ""
                guide_next = True
                curr_start = 0
                curr_end = min(curr_start+window_size, len(keywords))
                for j in range(number_of_words_per_sentence):
                    context, this_sequence, predicted_word, pred_word_stem = sample_sentence(context, this_sequence, orig_control_words, tokenizer, model, context_mask, keywords, encoded_keywords, curr_start, curr_end, converter_table, weight, guide_next, proba, only_max=only_max, upweight_control_words=upweight_control_words)
                    stemmed_curr_keywords = [re.sub('[^A-Za-z]+', ' ', porter.stem(g).lower().strip()).strip() for g in keywords]
                    stemmed_pred_word = re.sub('[^A-Za-z]+', ' ', pred_word_stem.lower().strip()).strip()
                    matching_index = -1
                    for k in range(curr_start,curr_end):
                        if stemmed_pred_word == stemmed_curr_keywords[k]: 
                            matching_index = k
                            break

                    if matching_index >= 0:
                        curr_start = matching_index+1
                        curr_end = min(curr_start+window_size, len(keywords))
                    logger.info(f"{curr_start} {curr_end} {keywords[curr_start:curr_end]} {context}")

                    context_mask.append(True)
                    if predicted_word == "<|endoftext|>":
                        break

                segments = context.split("__CUSTOMER__")
                cust_utt = segments[-1][0:segments[-1].index("<|endoftext|>")].strip() if "<|endoftext|>" in segments[
                    -1] else segments[-1].strip()
                context = " __CUSTOMER__ ".join(
                    segments[0:-1]).strip() + " __CUSTOMER__ " + cust_utt + " <|endoftext|> __AGENT__ "

                length = number_of_words_per_sentence
                perplexity = np.power(proba, (-1 / length))

                counter_sim = 0
                try:
                    if guidance_word is not None and len(guidance_word) > 0:
                        quality_score, word_count = evaluate_quality(this_sequence, guidance_word, counter_sim, perplexity,
                                                                 guide[b], temp)
                    else:
                        quality_score, word_count = 0, 0
                except:
                    import pdb; pdb.set_trace()

                result_subsequences.append(
                    [context, cum_quality_score[b] + quality_score, word_count, perplexity, proba, guidance_index[b],
                     guide[b], context_mask])

                perplexities[i] = perplexity

            ########################################################################################################
        # Deterministic DBS
        if not sample:
            result_subsequences_sorted = sorted(
                result_subsequences, key=lambda a_entry: a_entry[1], reverse=True)
            # Sample DBS
        else:
            scores = torch.tensor([a_entry[1] for a_entry in result_subsequences])
            soft_scores = F.softmax(scores, dim=-1)
            sampled_indeces = torch.multinomial(soft_scores, len(result_subsequences), replacement=False).tolist()
            result_subsequences_sorted = [result_subsequences[i] for i in sampled_indeces]

            del sampled_indeces
            del soft_scores
            del scores
            torch.cuda.empty_cache()

        # Select Beams
        for b in trange(number_of_beams, desc=f"Ordering beams for {word_index}"):
            full_text[b] = result_subsequences_sorted[b][0]
            full_mask[b] = result_subsequences_sorted[b][7]
            cum_quality_score[b] = result_subsequences_sorted[b][1]
            guidance_index[b] = result_subsequences_sorted[b][5]
            guide[b] = result_subsequences_sorted[b][6]
            if result_subsequences_sorted[b][2] > 0:  ## Word Count
                guidance_index[b] += 1
                if guidance_index[b] > number_keywords - 1:
                    guide[b] = False
                    guidance_index[b] = 0
                    success_length[b] = k + 1

            n_words_counter = (k + 1) * number_of_words_per_sentence
            online_probability[b] *= result_subsequences_sorted[b][4]
            online_perplexity = np.power(online_probability[b], (-1 / n_words_counter))

    #######################################
    # final evaluation
    #######################################
    for b in range(number_of_beams):
        if guide[b]:
            success_length[b] = 0

    # Success rate
    target_words = number_keywords
    target_count = 0
    for i in range(number_keywords):
        if count_word_stem(keywords[i], full_text[0]) > 0:
            target_count += 1

    if target_words > 0:
        success_rate = target_count / target_words
    else:
        success_rate = 0
    end_time = time.time()
    time_needed = end_time - start_time

    with open(final_text_file, "w") as final_f:
        final_f.write("Initial Context: ")
        final_f.write(first_utt + "\n\n")
        final_f.write(full_text[0] + "\n\n")
        final_f.write("keywords=")
        for w_index in range(len(keywords)):
            final_f.write(keywords[w_index])
            if w_index < len(keywords) - 1:
                final_f.write(",")
        final_f.write("\n")
        final_f.write("success_rate=" + str(success_rate) + "\n")
        final_f.write("success_length=" + str(success_length[0]) + "\n")
        final_f.write("time_needed=" + str(time_needed))
    # Time measurement

    processed_first_utt = tokenizer.decode(tokenizer.encode(first_utt))
    n_words_in_first_utt = len([w.strip() for w in processed_first_utt.split()])

    # declare evaluations
    evaluation = {
        "final_sequence: ": full_text[0],
        "generated_future: ": " ".join([w.strip() for w in full_text[0].split()][n_words_in_first_utt:]).split("<|endoftext|>")[0],
        "keywords": keywords,
        "success_rate": success_rate,
        "number_of_concurent_sentences": number_of_concurrent_sentences,
        "number_of_generated_sentences": number_of_generated_sentences,
        "number_of_words_per_sentence": number_of_words_per_sentence,
        "total_words": total_words,
        "top_k": top_k,
        "top_p": top_p,
        "constant": constant,
        "time_needed": time_needed,
        "success_length": success_length[0]
    }

    del model
    torch.cuda.empty_cache()

    return evaluation
