import argparse
import pickle
import torch
import os
from gpt import GPTLanguageModel
import configparser
from train_parameters import TrainParameter


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--conf', type=str, help="Setup Configuration Parameters")
    return parser.parse_args()


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.conf)
    params = TrainParameter(**config['options'])

    chars = ""
    with open(params.vocab_file, 'r', encoding='utf-8') as r:
        text = r.read()
        chars = sorted(list(set(text)))
    # Load Model
    model = GPTLanguageModel(chars, params)
    if os.path.isfile(params.pickle_file):
        with open(params.pickle_file, 'rb') as f:
            model = pickle.load(f)
        print("Model Loaded")
    while True:
        prompt = input('Prompt:\n')
        context = torch.tensor(model.encode(prompt), dtype=torch.long, device=model.get_device())
        generated_chars = model.decode(model.generate(context.unsqueeze(0),150, params.block_size)[0].tolist())
        print('Response: %s' % ''.join(generated_chars))

if __name__ == '__main__':
    main()