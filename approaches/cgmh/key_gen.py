import logging
import os

import torch
import torch.nn.functional
from tqdm import tqdm
from approaches.cgmh import reader
from approaches.cgmh.config import CGMHConfig
from approaches.cgmh.utils import *

config = CGMHConfig()

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("application.CGMH")


def get_probs(model, input_ids, context_ids, model_length=1024):
    # use the language model to calculate sentence probability
    with torch.no_grad():
        try:
            ext_input_ids = np.concatenate((np.tile(np.array(context_ids), (input_ids.shape[0], 1)), input_ids), axis=1)
            torch_input_ids = torch.tensor(ext_input_ids)[:, -model_length:].to(device)
            output = model(input_ids=torch_input_ids)
            logits = output.logits[:, len(context_ids):, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            cpu_numpy_probs = probs.detach().cpu().numpy()

        finally:
            if 'probs' in locals():
                del probs
            if 'logits' in locals():
                del logits
            if 'output' in locals():
                del output
            if 'torch_input_ids' in locals():
                del torch_input_ids

        return cpu_numpy_probs


def cgmh_generate(context, keywords, model, tokenizer, search_size=None):
    # CGMH sampling for key_gen
    mtest_forward = model
    id2sen = lambda ids: tokenizer.decode(ids, skip_special_tokens=True).split()
    context_ids = tokenizer.encode(context + ' ')

    # load keywords from file
    use_data, sta_vec_list = reader.prep_data_use(keywords, tokenizer, config.num_steps, config.key_num)
    for sen_id in tqdm(range(use_data.length)):
        # generate for each sequence of keywords
        sta_vec = sta_vec_list[sen_id % (config.num_steps - 1)]

        input, sequence_length, _ = use_data(1, sen_id)

        pos = 0
        outputs = []
        output_p = []
        for iter in tqdm(range(config.sample_time)):
            # ind is the index of the selected word, regardless of the beginning token.
            # sample config.sample_time times for each set of keywords
            config.sample_prior = [1, 10.0 / sequence_length[0], 1, 1]
            if iter % 20 < 10:
                config.threshold = 0
            else:
                config.threshold = 0.5
            ind = pos % (sequence_length[0])
            action = choose_action(config.action_prob)

            if sta_vec[ind] == 1 and action in [0, 2]:
                # skip words that we do not change(original keywords)
                action = 3

            # word replacement (action: 0)
            if action == 0 and ind < sequence_length[0] - 1:
                prob_old = get_probs(mtest_forward, input, context_ids)

                tem = 1
                for j in range(sequence_length[0] - 1):
                    tem *= prob_old[0][j][input[0][j + 1]]
                tem *= prob_old[0][j + 1][tokenizer.pad_token_id]
                prob_old_prob = tem

                input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(
                    input, sequence_length, ind, tokenizer, mode=action)
                prob_forward = get_probs(mtest_forward, input_forward, context_ids)[0, ind % (sequence_length[0] - 1), :]
                prob_mul = prob_forward
                input_candidate, sequence_length_candidate = generate_candidate_input(input, sequence_length,
                                                                                      ind, prob_mul,
                                                                                      search_size or config.search_size,
                                                                                      config.num_steps,
                                                                                      tokenizer,
                                                                                      mode=action)
                prob_candidate_pre = get_probs(mtest_forward, input_candidate, context_ids)
                prob_candidate = []
                for i in range(search_size or config.search_size):
                    tem = 1
                    for j in range(sequence_length[0] - 1):
                        tem *= prob_candidate_pre[i][j][input_candidate[i][j + 1]]
                    tem *= prob_candidate_pre[i][j + 1][tokenizer.pad_token_id]
                    prob_candidate.append(tem)

                prob_candidate = np.array(prob_candidate)
                prob_candidate_norm = normalize(prob_candidate)
                prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob = prob_candidate[prob_candidate_ind]
                if input_candidate[prob_candidate_ind][ind + 1] < len(tokenizer) - 2 and (
                        prob_candidate_prob > prob_old_prob * config.threshold or just_acc() == 0):
                    input = input_candidate[prob_candidate_ind:prob_candidate_ind + 1]
                pos += 1
                if ' '.join(id2sen(input[0])) not in output_p:
                    outputs.append([' '.join(id2sen(input[0])), prob_old_prob])

            # word insertion(action:1)
            if action == 1:
                if sequence_length[0] >= config.num_steps:
                    action = 3
                else:
                    input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(
                        input, sequence_length, ind, tokenizer, mode=action)
                    prob_forward = get_probs(mtest_forward, input_forward, context_ids)[0, ind % (sequence_length[0] - 1), :]
                    prob_mul = prob_forward
                    input_candidate, sequence_length_candidate = generate_candidate_input(input,
                                                                                          sequence_length, ind,
                                                                                          prob_mul,
                                                                                          search_size or config.search_size,
                                                                                          config.num_steps,
                                                                                          tokenizer,
                                                                                          mode=action)
                    prob_candidate_pre = get_probs(mtest_forward, input_candidate, context_ids)

                    prob_candidate = []
                    for i in range(search_size or config.search_size):
                        tem = 1
                        for j in range(sequence_length_candidate[0] - 1):
                            tem *= prob_candidate_pre[i][j][input_candidate[i][j + 1]]
                        tem *= prob_candidate_pre[i][j + 1][tokenizer.pad_token_id]
                        prob_candidate.append(tem)
                    prob_candidate = np.array(prob_candidate) * config.sample_prior[1]
                    prob_candidate_norm = normalize(prob_candidate)

                    prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
                    prob_candidate_prob = prob_candidate[prob_candidate_ind]

                    prob_old = get_probs(mtest_forward, input, context_ids)

                    tem = 1
                    for j in range(sequence_length[0] - 1):
                        tem *= prob_old[0][j][input[0][j + 1]]
                    tem *= prob_old[0][j + 1][tokenizer.pad_token_id]

                    prob_old_prob = tem
                    # alpha is acceptance ratio of current proposal
                    alpha = min(1, prob_candidate_prob * config.action_prob[2] / (
                            prob_old_prob * config.action_prob[1] * prob_candidate_norm[
                        prob_candidate_ind]))
                    if ' '.join(id2sen(input[0])) not in output_p:
                        outputs.append([' '.join(id2sen(input[0])), prob_old_prob])
                    if choose_action([alpha, 1 - alpha]) == 0 and input_candidate[prob_candidate_ind][
                        ind + 1] < len(tokenizer) - 2 and (
                            prob_candidate_prob > prob_old_prob * config.threshold or just_acc() == 0):
                        input = input_candidate[prob_candidate_ind:prob_candidate_ind + 1]
                        sequence_length += 1
                        pos += 2
                        sta_vec.insert(ind, 0.0)
                        del (sta_vec[-1])
                    else:
                        action = 3

            # word deletion(action: 2)
            if action == 2 and ind < sequence_length[0] - 1:
                if sequence_length[0] <= 2:
                    action = 3
                else:

                    prob_old = get_probs(mtest_forward, input, context_ids)

                    tem = 1
                    for j in range(sequence_length[0] - 1):
                        tem *= prob_old[0][j][input[0][j + 1]]
                    tem *= prob_old[0][j + 1][tokenizer.pad_token_id]
                    prob_old_prob = tem
                    input_candidate, sequence_length_candidate = generate_candidate_input(input,
                                                                                          sequence_length, ind,
                                                                                          None,
                                                                                          search_size or config.search_size,
                                                                                          config.num_steps,
                                                                                          tokenizer,
                                                                                          mode=2)
                    prob_new = get_probs(mtest_forward, input_candidate, context_ids)
                    tem = 1
                    for j in range(sequence_length_candidate[0] - 1):
                        tem *= prob_new[0][j][input_candidate[0][j + 1]]
                    tem *= prob_new[0][j + 1][tokenizer.pad_token_id]
                    prob_new_prob = tem

                    input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(
                        input, sequence_length, ind, tokenizer, mode=0)
                    prob_forward = get_probs(mtest_forward, input_forward, context_ids)[0, ind % (sequence_length[0] - 1), :]
                    prob_mul = prob_forward
                    input_candidate, sequence_length_candidate = generate_candidate_input(input,
                                                                                          sequence_length, ind,
                                                                                          prob_mul,
                                                                                          search_size or config.search_size,
                                                                                          config.num_steps,
                                                                                          tokenizer,
                                                                                          mode=0)
                    prob_candidate_pre = get_probs(mtest_forward, input_candidate, context_ids)

                    prob_candidate = []
                    for i in range(search_size or config.search_size):
                        tem = 1
                        for j in range(sequence_length[0] - 1):
                            tem *= prob_candidate_pre[i][j][input_candidate[i][j + 1]]
                        tem *= prob_candidate_pre[i][j + 1][tokenizer.pad_token_id]
                        prob_candidate.append(tem)
                    prob_candidate = np.array(prob_candidate)

                    # alpha is acceptance ratio of current proposal
                    prob_candidate_norm = normalize(prob_candidate)
                    if input[0] in input_candidate:
                        for candidate_ind in range(len(input_candidate)):
                            if input[0] in input_candidate[candidate_ind: candidate_ind + 1]:
                                break
                            pass
                        alpha = min(
                            prob_candidate_norm[candidate_ind] * prob_new_prob * config.action_prob[1] / (
                                    config.action_prob[2] * prob_old_prob), 1)
                    else:
                        pass
                        alpha = 0
                    if ' '.join(id2sen(input[0])) not in output_p:
                        outputs.append([' '.join(id2sen(input[0])), prob_old_prob])
                    if choose_action([alpha, 1 - alpha]) == 0 and (
                            prob_new_prob > prob_old_prob * config.threshold or just_acc() == 0):
                        input = np.concatenate(
                            [input[:, :ind + 1], input[:, ind + 2:], input[:, :1] * 0 + tokenizer.pad_token_id],
                            axis=1)
                        sequence_length -= 1
                        pos += 0
                        del (sta_vec[ind])
                        sta_vec.append(0)
                    else:
                        action = 3
            # skip word (action: 3)
            if action == 3:
                pos += 1
            if outputs != []:
                output_p.append(outputs[-1][0])

        # choose output from samples
        outputss = None
        for num in range(config.min_length, 0, -1):
            outputss = [x for x in outputs if len(x[0].split()) >= num]
            if outputss != []:
                break
        if outputss == []:
            outputss.append([' '.join(id2sen(input[0])), 1])
        outputss = sorted(outputss, key=lambda x: x[1])[::-1]

        return outputss[0][0]
