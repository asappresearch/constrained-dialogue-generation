import random
import re
from collections import Counter

import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

porter = PorterStemmer()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_train_config(config):
    assert config.has_section("TRAINING")
    assert config.has_option('TRAINING', 'model_type')
    assert config.get('TRAINING', 'model_type', fallback=None) is not None

    assert config.has_option('DATALOADER', 'block_size')
    assert config.get('DATALOADER', 'block_size', fallback=None) is not None

    if config.has_option("DATALOADER", "issueid_to_intent_file"):
      assert config.get('DATALOADER', 'issueid_to_intent_file', fallback=None) is not None

    assert config.has_option("TRAINING", "output_dir")
    assert config.get('TRAINING', 'output_dir', fallback=None) is not None

    assert config.has_option("TRAINING", "per_gpu_train_batch_size")
    assert config.get('TRAINING', 'per_gpu_train_batch_size', fallback=None) is not None

    assert config.has_option("TRAINING", "batch_size")
    assert config.get('TRAINING', 'batch_size', fallback=None) is not None

    assert config.has_option("TRAINING", "max_steps")
    assert config.get('TRAINING', 'max_steps', fallback=None) is not None

    assert config.has_option("TRAINING", "num_train_epochs")
    assert config.get('TRAINING', 'num_train_epochs', fallback=None) is not None

    assert config.has_option("TRAINING", "gradient_accumulation_steps")
    assert config.get('TRAINING', 'gradient_accumulation_steps', fallback=None) is not None

    assert config.has_option("TRAINING", "weight_decay")
    assert config.get('TRAINING', 'weight_decay', fallback=None) is not None

    assert config.has_option("TRAINING", "learning_rate")
    assert config.get('TRAINING', 'learning_rate', fallback=None) is not None

    assert config.has_option("TRAINING", "adam_epsilon")
    assert config.get('TRAINING', 'adam_epsilon', fallback=None) is not None

    assert config.has_option("TRAINING", "warmup_steps")
    assert config.get('TRAINING', 'warmup_steps', fallback=None) is not None

    assert config.has_option("TRAINING", "max_grad_norm")
    assert config.get('TRAINING', 'max_grad_norm', fallback=None) is not None

    assert config.has_option("TRAINING", "logging_steps")
    assert config.get('TRAINING', 'logging_steps', fallback=None) is not None

    assert config.has_option("TRAINING", "save_steps")
    assert config.get('TRAINING', 'save_steps', fallback=None) is not None

def remove_stopwords(s):
    lowered = re.sub('[^A-Za-z]+', ' ', s.lower())
    stop_words = set(stopwords.words('english') + ["information","submitted", "delayed", "digits", "ok", "okay", "thank", "thanks", "you", "yes", "would", "like", "get", "much", "really", "hello", "yeah", "oh", "hi"])
    word_tokens = word_tokenize(lowered)  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_conv_ngrams(cust_utts, ngram=1):
    ngrams = []
    for c in cust_utts:
        ngrams.extend([n for n in find_ngrams(c.split(), ngram)])
    keywords = " ".join([x[0][0] for x in Counter(ngrams).most_common(5)])
    return keywords

def get_preprocessed_cust_utt(conv, tokenizer=None):
    if isinstance(conv, dict):
        cust_utts = []
        for text, spkr in zip(conv['text'], conv['spkr']):
            if spkr == "customer":
                cust_utts.append(text)
    else:
        cust_utts = [t.strip() for t in conv.split("__CUSTOMER__") if len(t) > 0]

    dataset = []
    eos_token = tokenizer.eos_token if tokenizer is not None else "<|endoftext|>"
    cust_utts = [t[:t.index(eos_token)] if eos_token in t else t for t in cust_utts]
    preprocessed = [remove_stopwords(c) for c in cust_utts]
    preprocessed = [c for c in preprocessed if len(c)>0]
    return preprocessed

def get_tfidf_topn(conv, vectorizer, n=20):
    feature_array = np.array(vectorizer.get_feature_names())
    keywords = []
    tfidf_out = vectorizer.transform([conv])
    tfidf_sorting = np.argsort(tfidf_out.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]
    keywords = " ".join(top_n)
    return keywords

def get_tfidf_vectorizer(corpus, min_df=5):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df)
    vectorizer.fit(corpus)
    return vectorizer

def get_cust_utts_future(example, start_index):
    cust_utts = " ".join([f"{utt}" for i, (utt, spkr) in enumerate(zip(example["text"], example["spkr"])) if i >= start_index and spkr == "customer"])
    return re.sub('[^A-Za-z]+', ' ', cust_utts.lower())

def get_cust_utts_past(example, stop_index):
    cust_utts = " ".join([f"{utt}" for i, (utt, spkr) in enumerate(zip(example["text"], example["spkr"])) if i < stop_index and spkr == "customer"])
    return re.sub('[^A-Za-z]+', ' ', cust_utts.lower())

def count_word_stem(word, sequence):
    cust_utts = sequence
    sequence = cust_utts.split()
    word_count = 0

    word_stem = porter.stem(word).lower().strip()

    for s_word in sequence:
        s_word_stem = porter.stem(s_word)
        if(s_word_stem.lower().strip() == word_stem):
            word_count += 1

    return word_count

def count_keywords(future_text, past_text, all_keywords):
    past_text = re.sub('[^A-Za-z]+', ' ', past_text.lower())
    future_text = re.sub('[^A-Za-z]+', ' ', future_text.lower())
    keywords = [k for k in all_keywords if k not in past_text]
    completed_keywords = [k for k in all_keywords if k in past_text]
    num_keywords = 0
    observed_keywords, unobserved_keywords = [], []
    for keyword in keywords:
        if count_word_stem(keyword, future_text) > 0:
            num_keywords += 1
            if keyword not in observed_keywords:
                observed_keywords.append(keyword)
    unobserved_keywords = [k for k in keywords if k not in observed_keywords]
    if len(keywords)>0:
        return num_keywords/len(keywords), completed_keywords, keywords, observed_keywords
    return 0, completed_keywords, keywords, observed_keywords

def get_config_template():
    template = {
        'DEFAULT': {
                        'log_level': '',
                        'local_rank': '',
                        'seed': '',
                        'train_task': '',
                        'no_cuda': ''
        },
        'DATALOADER': {
                        'block_size': '',
                        'prompt_type': '',
                        'keyword_length': '',
                        'num_keywords': '',
                        'eval_data_file': '',
                        'train_data_file': '',
                        'cache_file_postfix': '',
                        'keywords_dir': '',
                        'issueid_to_intent_file': '',
                        'kvstore_n_traincontexts': '',
                        'kvstore_n_testcontexts': '',
                        'eval_future_type': ''
        },
        'TRAINING': {
            'model_type': '',
            'model_class': '',
            'output_dir': '',
            'per_gpu_train_batch_size': '',
            'batch_size': '',
            'max_steps': '',
            'num_train_epochs': '',
            'gradient_accumulation_steps': '',
            'weight_decay': '',
            'learning_rate': '',
            'adam_epsilon': '',
            'warmup_steps': '',
            'fp16_opt_level': '',
            'fp16': '',
            'block_method': '',
            'max_grad_norm': '',
            'logging_steps': '',
            'additional_tokens_path': '',
            'save_steps': '',
            'eval_model_path': '',
            'eval_output_dir': '',
            'per_gpu_eval_batch_size': '',
            'evaluate_all_checkpoints': ''
        },
        'DBS': {
            'n_concurrent_sentences': '',
            'n_generated_sentences': '',
            'n_beams': '',
            'output_file': '',
            'n_keywords': '',
            'embedding_type': '',
            'split': '',
            'agent_model_path': '',
        }
    }

    return template
