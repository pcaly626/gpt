from dataclasses import dataclass


@dataclass
class TrainParameter:

    def __init__(self, block_size:int, batch_size:int, n_layer:int, n_head: int, n_embd:int, 
                 eval_iters:int, max_iters:int, learning_rate:float, dropout:float, train_file:str,
                 val_file:str, pickle_file:str, vocab_file:str):
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.eval_iters = int(eval_iters)
        self.max_iters = int(max_iters)
        self.learning_rate = float(learning_rate)
        self.dropout = float(dropout)
        self.train_file = train_file
        self.val_file = val_file
        self.vocab_file = vocab_file
        self.pickle_file = pickle_file

    block_size: int
    batch_size: int
    n_layer: int
    n_head: int 
    n_embd: int
    eval_iters: int
    max_iters: int
    learning_rate: float
    dropout: float
    train_file: str
    val_file: str
    pickle_file: str
    vocab_file: str