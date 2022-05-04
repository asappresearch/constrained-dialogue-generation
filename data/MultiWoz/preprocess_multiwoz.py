import os, sys
sys.path.append("../../")
import json
from argparse import ArgumentParser
from tqdm import tqdm as progress_bar
from approaches.helpers import set_seed

def main(args):
    domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
    splits = ['train', 'dev', 'test']

    data = json.load(open(f"data.json", 'r'))
    ont = json.load(open(f"ontology.json", 'r'))

    with open('valListFile.json', 'r') as valfile:
        val_list = [line.rstrip('\n') for line in valfile]
    with open('testListFile.json', 'r') as testfile:
        test_list = [line.rstrip('\n') for line in testfile]
    valid_ont = {domain: {} for domain in domains}

    size = len(data)
    print('data size', len(data))

    for domain_slot, values in ont.items():
        domain, slot = domain_slot.split('-')
        if domain in domains and len(values) > 2 and len(values) < 100:
            valid_ont[domain][slot] = values

    conv_id_counter = 0
    convid_map = {}
    final = {split: [] for split in splits}
    for guid, conversation in progress_bar(data.items(), total=size):
        speaker = 'customer'

        if guid in val_list:
            split = 'dev'
        elif guid in test_list:
            split = 'test'
        else:
            split = 'train'

        topics = []
        for domain, slot_vals in conversation['goal'].items():
            if domain in domains and len(slot_vals) > 0:
                topics.append(domain)

        new_convo = {'convo_id': conv_id_counter, "original": [], 'topics': topics}
        convid_map[guid] = conv_id_counter
        conv_id_counter += 1
        for turn in conversation['log']:
            if turn['turn_id'] % 2 == 0:
                speaker = 'customer'
            else:
                speaker = 'agent'
            new_turn = [speaker, turn['text']]
            new_convo['original'].append(new_turn)

        final[split].append(new_convo)

    for split, processed in final.items():
        print(split, len(processed))

    json.dump(final, open(args.output_file, 'w'))

    with open('convid_map.json', "w") as f:
        json.dump(convid_map, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-file")
    parser.add_argument("--seed", default=1234, type=int)
    arguments = parser.parse_args()
    set_seed(arguments.seed)
    main(arguments)
