from copy import copy

import numpy as np


def normalize(x, e=0.05):
    tem = copy(x)
    if max(tem) == 0:
        tem += e
    return tem / tem.sum()


def cut_from_point(input_, sequence_length, ind, tokenizer, mode=0):
    batch_size = input_.shape[0]
    num_steps = input_.shape[1]
    input_forward = np.zeros([batch_size, num_steps]) + tokenizer.pad_token_id
    input_backward = np.zeros([batch_size, num_steps]) + tokenizer.pad_token_id
    sequence_length_forward = np.zeros([batch_size])
    sequence_length_backward = np.zeros([batch_size])
    for i in range(batch_size):
        input_forward[i][0] = tokenizer.encode('__CUSTOMER__')[0]
        input_backward[i][0] = tokenizer.encode('__CUSTOMER__')[0]
        length = sequence_length[i] - 1

        for j in range(ind):
            input_forward[i][j + 1] = input_[i][j + 1]
        sequence_length_forward[i] = ind + 1
        if mode == 0:
            for j in range(length - ind - 1):
                input_backward[i][j + 1] = input_[i][length - j]
            sequence_length_backward[i] = length - ind
        elif mode == 1:
            for j in range(length - ind):
                input_backward[i][j + 1] = input_[i][length - j]
            sequence_length_backward[i] = length - ind + 1
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(
        np.int32), sequence_length_backward.astype(np.int32)


def generate_candidate_input(input_, sequence_length, ind, prob, search_size, num_steps, tokenizer, mode=0):
    input_new = np.array([input_[0]] * search_size)
    sequence_length_new = np.array([sequence_length[0]] * search_size)
    ind_token = None
    if mode != 2:
        ind_token = np.argsort(prob[:])[-search_size:]

    if mode == 2:
        for i in range(sequence_length[0] - ind - 2):
            input_new[:, ind + i + 1] = input_new[:, ind + i + 2]
        for i in range(sequence_length[0] - 1, num_steps - 1):
            input_new[:, i] = input_new[:, i] * 0 + tokenizer.pad_token_id
        sequence_length_new = sequence_length_new - 1
        return input_new[:1], sequence_length_new[:1]
    if mode == 1:
        for i in range(0, sequence_length_new[0] - 1 - ind):
            input_new[:, sequence_length_new[0] - i] = input_new[:, sequence_length_new[0] - 1 - i]
        sequence_length_new = sequence_length_new + 1
    for i in range(search_size):
        input_new[i][ind + 1] = ind_token[i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)


def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i] + c[i - 1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def just_acc():
    r = np.random.random()
    if r < 0.0:
        return 0
    else:
        return 1
