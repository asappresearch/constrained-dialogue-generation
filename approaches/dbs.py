import logging
import math
import os
import time
import numpy as np
import torch
from tqdm import trange

os.environ[
    'TRANSFORMERS_CACHE'] = './transformer_models'  # Work-around to avoid memory problems in server, comment out depending on memory availability

import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from approaches.helpers import set_seed

porter = PorterStemmer()
IGNORE_INDEX = -100
import gensim.downloader as api
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("application.DBS")

def glove_encode(glove_encoder, word):
    return glove_encoder(word)


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
        import gensim.downloader as api
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

    word_stem = porter.stem(word).lower().strip()

    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        if (s_word_stem.lower().strip() == word_stem):
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
        return quality_score, 0

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


def sample_sentence(text, this_sequence, tokenizer, model, mask, guide_word_stem, glove_word, converter_table, weight,
                    guide=False, prev_proba=1, top_k=0, top_p=0.9, temperature=1., only_max=False):
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

    logits = outputs.logits
    logits = logits[0, -1, :] / temperature
    proba = F.softmax(logits, dim=-1)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Calculate cosine similarity
    if guide == True and glove_word is not None:
        sim = cosine_similarity(np.reshape(glove_word, (1, -1)), converter_table)
        if only_max == True:
            sim_aux = np.zeros_like(sim)
            sim_aux[0, sim.argmax()] = sim.max()
            sim = sim_aux
        else:
            sim = np.clip(np.squeeze(sim), a_min=0, a_max=None)
        sim = sim * sim
        weighted_sim = torch.tensor(sim * weight).to(DEVICE)
        logits = logits + weighted_sim
        del weighted_sim
        del sim

    logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()

    predicted_word = tokenizer.decode([predicted_index])
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]

    pred_word_stem = porter.stem(pred_word)

    if pred_word_stem.lower().strip() == guide_word_stem.lower().strip():
        guide_next = False
    else:
        guide_next = guide
    next_prob = prev_proba * proba[predicted_index].item()
    del logits
    del proba
    torch.cuda.empty_cache()
    return predicted_text, guide_next, next_prob, this_sequence, predicted_word


def sample_sentence_noguide(text, this_sequence, tokenizer, model, mask, prev_proba=1, top_k=0, top_p=0.9,
                            temperature=1.):
    """ Samples the next word of the sequence without logit modification (guidance
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
        pdb.set_trace()
    del tokens_tensor

    torch.cuda.empty_cache()

    logits = outputs.logits
    logits = logits[0, -1, :] / temperature
    proba = F.softmax(logits, dim=-1)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()

    predicted_word = tokenizer.decode([predicted_index])
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]
    next_prob = prev_proba * proba[predicted_index].item()
    del logits
    del proba
    torch.cuda.empty_cache()
    return predicted_text, next_prob, this_sequence, predicted_word


def encode_keywords(embedding_type, model, tokenizer, word_index, index_keywords, save_dir):
    save_path = os.path.join(save_dir, f"{embedding_type}_set_{word_index}.npy")
    if word_index is not None:
        if os.path.exists(save_path):
            encoded_words = np.load(save_path)
            return encoded_words
    logger.info(f"Generating {embedding_type} embeddings for keywords...")
    if embedding_type == "glove":
        encoder = api.load("glove-wiki-gigaword-300")
    elif embedding_type == "lm":
        encoder = model.transformer.wte.weight
    encoded_words = []
    for word in index_keywords:
        if embedding_type == "glove":
            if word in encoder:
                embedding = encoder[word]
                logger.info(f"encoder {max(embedding)} {min(embedding)}")
            else:
                embedding = np.random.uniform(low=-2, high=2, size=(300,))
                logger.info(f"random {max(embedding)} {min(embedding)}")
        elif embedding_type == "lm":
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

        # TODO: If we want to have a Pegasus agent then this should be dragged out
        #  to a method that can load any type of model

    # Get encoded vocabulary
    converter_table = converter_embedding_table(embedding_type, model, tokenizer, save_path)
    # Get encoded keywords
    if len(index_keywords)>0:
        encoded_keywords = encode_keywords(embedding_type, model, tokenizer, word_index, index_keywords, save_dir=save_path)
    else:
        encoded_keywords = None
    keywords = index_keywords[0:num_keywords]
    logger.info(f"Keywords: {keywords}")

    # Change full_text to desired initial context
    full_text = [f"{first_utt} "] * number_of_beams
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

    for k in range(number_of_generated_sentences):
        # Define guidance word and index for guidance in model and quality function
        result_subsequences = []
        for b in range(number_of_beams):
            # Reset variables:
            beam_text = full_text[b]
            beam_mask = full_mask[b]
            if guidance_index[b] < len(keywords):
                guidance_word = keywords[guidance_index[b]]
                guide_word_stem = porter.stem(guidance_word)
            else:
                guidance_word, guide_word_stem = "", ""
            perplexities = np.zeros((number_of_concurrent_sentences))

            ####################################### Generation loop ################################################
            for i in range(number_of_concurrent_sentences):
                context = beam_text
                context_mask = list(beam_mask)
                proba = 1
                this_sequence = ""
                if guide[b]:
                    guide_next = True
                    for j in range(number_of_words_per_sentence):
                        if encoded_keywords is None or b>=len(guidance_index) or guidance_index[b] >= len(encoded_keywords):
                            glove_word = None
                        else:
                            glove_word = encoded_keywords[guidance_index[b]]

                        if weight is None:
                            raise ValueError("lambda parameter cannot be None for DBS approaches")

                        context, guide_next, proba, this_sequence, predicted_word = sample_sentence(context,
                                                                                                    this_sequence,
                                                                                                    tokenizer, model,
                                                                                                    context_mask,
                                                                                                    guide_word_stem,
                                                                                                    glove_word,
                                                                                                    converter_table,
                                                                                                    weight, guide_next,
                                                                                                    proba,
                                                                                                    only_max=only_max)
                        context_mask.append(True)
                        if predicted_word == "<|endoftext|>":
                            break
                else:  # Dont't guide
                    for j in range(number_of_words_per_sentence):
                        context, proba, this_sequence, predicted_word = sample_sentence_noguide(context, this_sequence,
                                                                                                tokenizer, model,
                                                                                                context_mask,
                                                                                                prev_proba=proba)
                        context_mask.append(True)
                        if predicted_word == "<|endoftext|>":
                            break

                segments = context.split("__CUSTOMER__")
                cust_utt = segments[-1][0:segments[-1].index("<|endoftext|>")].strip() if "<|endoftext|>" in segments[
                    -1] else segments[-1].strip()
                context = " __CUSTOMER__ ".join(
                    segments[0:-1]).strip() + " __CUSTOMER__ " + cust_utt + " <|endoftext|> __AGENT__ "

                if number_of_generated_sentences > 1:
                    agent_utt = Generation.generate_all(context,
                                                        models,
                                                        tokenizers,
                                                        num_timesteps=1,
                                                        num_convs=1,
                                                        customer_model_type="GPT2",
                                                        first_speaker="agent")

                    agent_utt = agent_utt[0:agent_utt.index("<|endoftext|>")].strip() if "<|endoftext|>" in agent_utt else \
                        agent_utt.strip()

                    logger.info(f"Context: {context}")
                    logger.info(f"Agent Utt: {agent_utt}")

                    agent_utt_tokens = agent_tokenizer.encode(agent_utt)
                    context = context + agent_utt + " <|endoftext|> __CUSTOMER__ "
                    context_mask += [False] + [False]*len(agent_utt_tokens) + [False]

                length = number_of_words_per_sentence
                perplexity = np.power(proba, (-1 / length))

                counter_sim = 0
                quality_score, word_count = evaluate_quality(this_sequence, guidance_word, counter_sim, perplexity,
                                                             guide[b], temp)

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

        ''' Uncomment to write all intermediate steps to .txt

        text_file.write("\nBest 10 next subsequences: \n")
        for result_subsequence in result_subsequences_sorted:
            text_file.write(result_subsequence[0] + "\n Perplexity:" +
                            str(result_subsequence[2]) + "\n Quality Score: " +
                            str(result_subsequence[1]) + "\n\n")

        text_file.write("\n\n\n\n")
        '''
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
