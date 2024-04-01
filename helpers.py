import torch
import mmap
import random


def get_random_chunk(model, filename, block_size, batch_size):
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r','')

            # Train and test splits
            data = torch.tensor(model.encode(decoded_block), dtype=torch.long)
    return data


def get_batch(model, filename, block_size, batch_size):
    data = get_random_chunk(model, filename, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(model.get_device()), y.to(model.get_device())


@torch.no_grad()
def estimate_loss(split, model, eval_iters, block_size, batch_size):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for key in split.keys():
        for k in range(eval_iters):
            X, Y  = get_batch(model, split[key], block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[key] = losses.mean()
        model.train()
    return out
