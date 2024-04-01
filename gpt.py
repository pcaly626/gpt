import torch
import torch.nn as nn
import torch.nn.functional as F


# GPT 
class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape  # Batch, Time, Channel
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T) Swaps last with second to last
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        '''
        [1,-inf,-inf]
        [1,.4,-inf]
        [1,.4,.6]
        '''
        wei = F.softmax(wei, dim=-1)
        '''
        [1,0,0]
        [1,.4,0]
        [1,.4,.6]
        '''
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
        

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # Looks at a num, if its <=0 make it 0, else leave it alone
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)  # Makes a certain % of neurons become 0 so we dont over fit
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    '''
    Transformer Block
    '''
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab, params):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(vocab), params.n_embd)  # Each Character
        self.position_embedding_table = nn.Embedding(params.block_size, params.n_embd) # Each Embedding
        block = [Block(params.n_embd, params.n_head, params.block_size, params.dropout) for _ in range(params.n_layer)]
        self.blocks = nn.Sequential(*block)  # Decoder blocks

        self.ln_f = nn.LayerNorm(params.n_embd)
        self.lm_head = nn.Linear(params.n_embd, len(vocab))
        self.apply(self._init_weights)
        self.to(self.get_device())
        self.string_to_int = {ch:i for i,ch in enumerate(vocab)}
        self.int_to_string = {i:ch for i,ch in enumerate(vocab)}
        # self.loss = None

    def encode(self, s):
        encoding = []
        for c in s:
            item = c
            if type(c) == torch.Tensor:
                item = c.item()
            encoding.append(self.string_to_int[item])
        return encoding
    
    def decode(self, l):
        decoding = []
        for i in l:
            item = i
            if type(i) == torch.Tensor:
                item = i.item()
            decoding.append(self.int_to_string[item])
        return decoding
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def stop_generation(self, index, max_new_tokens):
        if index.size()[1] >= max_new_tokens and self.decode(index[0][-1:])[0] in ['.', '!','?','".',"'."]:
            return False
        return True

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else 'cpu'

    def forward(self, index, targets=None):
        B,T = index.shape
        # idx and targers are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.get_device()))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            # Batch, TIME (sequence of ints), Channel (vocab size)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # if loss <= self.loss:  # Take the lower logits instead of the new higher logits
            #     self.loss = loss
            #     return logits, loss
            # else:
            #     return index, loss
            
        return logits, loss

    def generate(self, index, max_new_tokens, block_size):
        # index is (B, T) array of indices in the current context
        while self.stop_generation(index, max_new_tokens):
            # crop index to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
