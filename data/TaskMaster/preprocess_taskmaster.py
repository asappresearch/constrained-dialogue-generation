import os, sys
sys.path.append("../../")
from argparse import ArgumentParser
import json
import random
from tqdm import tqdm
from approaches.helpers import set_seed

def main(args):
    processed_data = []
    conv_id_counter = 0
    conv_id_map = {}
    for fname in tqdm(os.listdir(args.data_dir)):
        if '.json' not in fname:
            continue

        filename = os.path.join(args.data_dir, fname)

        with open(filename, 'r') as f:
            raw_data = json.load(f)

        for obj in raw_data:
            conv_id = obj['conversation_id']
            original_obj = []
            for utt in obj['utterances']:
                spkr = None
                if utt['speaker'] == 'user':
                    spkr = 'customer'
                elif utt['speaker'] == 'assistant':
                    spkr = 'agent'
                else:
                    raise ValueError(f'unrecognized speaker {utt["speaker"]}')

                original_obj.append([spkr, utt['text']])
            processed_data.append({"convo_id": conv_id_counter, "original": original_obj})
            conv_id_map[conv_id] = conv_id_counter
            conv_id_counter += 1

    all_idxs = [i for i in range(len(processed_data))]
    random.shuffle(all_idxs)

    cutoff = int(len(all_idxs) * 0.7)
    train_idxs = all_idxs[:cutoff]
    remaining = all_idxs[cutoff:]
    dev_idxs = remaining[:int(len(remaining)/2)]
    test_idxs = remaining[int(len(remaining)/2):]

    final_output = {'train': [], 'dev': [], 'test': []}
    for idx in train_idxs:
        final_output['train'].append(processed_data[idx])

    for idx in dev_idxs:
        final_output['dev'].append(processed_data[idx])

    for idx in test_idxs:
        final_output['test'].append(processed_data[idx])

    with open(args.output_file, "w") as f:
        json.dump(final_output, f)

    with open('convid_map.json', "w") as f:
        json.dump(conv_id_map, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-file")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--seed", default=1234)
    arguments = parser.parse_args()
    set_seed(arguments.seed)
    main(arguments)
