import numpy as np


class Dataset:
    def __init__(self, input_, sequence_length, target):
        self.input = input_
        self.target = target
        self.sequence_length = sequence_length
        self.length = len(input_)

    def __call__(self, batch_size, step):
        batch_num = self.length // batch_size
        step = step % batch_num
        return self.input[step * batch_size: (step + 1) * batch_size], \
            self.sequence_length[step * batch_size: (step + 1) * batch_size], \
            self.target[step * batch_size: (step + 1) * batch_size]


def prep_data_use(keywords, tokenizer, max_length, key_num):
    data = []
    sta_vec_list = []

    sta_vec = list(np.zeros([max_length - 1]))
    key = choose_key(keywords, key_num)
    for i in range(len(key)):
        sta_vec[i] = 1
    sta_vec_list.append(sta_vec)
    data.append(tokenizer.encode(' '.join(key)))

    data_new = array_data(data, max_length, tokenizer)
    return data_new, sta_vec_list


def choose_key(line, num):
    ind_list = list(range(len(line)))
    np.random.shuffle(ind_list)
    ind_list = ind_list[:num]
    ind_list.sort()
    tem = []
    for ind in ind_list:
        tem.append(line[ind])
    return tem


def array_data(data, max_length, tokenizer, shuffle=False):
    max_length_m1 = max_length - 1
    if shuffle:
        np.random.shuffle(data)
    sequence_length_pre = np.array([len(line) for line in data]).astype(np.int32)
    sequence_length = []
    for item in sequence_length_pre:
        if item > max_length_m1:
            sequence_length.append(max_length)
        else:
            sequence_length.append(item + 1)
    sequence_length = np.array(sequence_length)
    for i in range(len(data)):
        if len(data[i]) >= max_length_m1:
            data[i] = data[i][:max_length_m1]
        else:
            for j in range(max_length_m1 - len(data[i])):
                data[i].append(tokenizer.pad_token_id)
        data[i].append(tokenizer.pad_token_id)
    target = np.array(data).astype(np.int32)
    input_ = np.concatenate([np.ones([len(data), 1]) * tokenizer.encode('__CUSTOMER__'),
                             target[:, :-1]], axis=1).astype(np.int32)

    return Dataset(input_, sequence_length, target)
