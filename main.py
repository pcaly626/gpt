import argparse
import pickle
import torch
import os
from gpt2 import GPTLanguageModel
from helpers import estimate_loss, get_batch
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
    # Get Vocab
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

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    
    split = {'train': params.train_file, 'val': params.val_file}
    
    # Training Loop
    for iter in range(params.max_iters):
        if iter % params.eval_iters == 0 and iter != 0:
            losses = estimate_loss(split, model, params.eval_iters, params.block_size, params.batch_size)
            print(f"step: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")
        # sample a batch of data
        xb, yb = get_batch(model, split['train'], params.block_size, params.batch_size)
        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save Model
    with open(params.pickle_file, 'wb') as f:
        pickle.dump(model, f)
    print('Model Saved')

if __name__ == '__main__':
    main()