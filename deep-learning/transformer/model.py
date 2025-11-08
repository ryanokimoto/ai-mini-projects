from torch import nn
import torch
import math

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :]


"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""
class Transformer(nn.Module):
    def __init__(self, vocab_size, d, num_heads, num_layers, d_ff=None, max_seq_len=256, dropout=0.1): # V: vocab size, d: model dimension
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d
        self.d = d
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d)
        self.position_encoding = SinusoidalPositions(max_seq_len, d)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size)


    def forward(self, x, attention_mask=None):
        x = self.embeddings(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        logits = self.out(x)
        return logits

class Block(nn.Module):
    def __init__(self, d, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d)
        self.drop_sa = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d)
        self.drop_ff = nn.Dropout(dropout)
        self.sa = SA(d, num_heads)
        self.ff = FFN(d, d_ff)


    def forward(self, x, attention_mask=None):
        sa_x = self.norm_sa(x)
        sa_x = self.sa(sa_x, attention_mask)
        sa_x = self.drop_sa(sa_x)
        x = x + sa_x

        ff_x = self.norm_ff(x)
        ff_x = self.ff(ff_x)
        ff_x = self.drop_ff(ff_x)
        x = x + ff_x
        return x

class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class SA(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.Q = nn.Linear(d, d)
        self.K = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x, attention_mask=None):
        B, S, d = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scaled_dots = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scaled_dots = self.mask(scaled_dots, attention_mask)
        scores = torch.softmax(scaled_dots, dim=-1)

        x = scores @ v
        x = x.transpose(1, 2).contiguous().view(B, S, d)

        # scaled_dots = q @ k / math.sqrt(q.size(-1))
        # scaled_dots = scaled_dots + self.mask(scaled_dots, attention_mask)
        # scores = torch.softmax(scaled_dots, dim=-1)

        return self.out(x)

    def mask(self, dots, attention_mask):
        B, num_heads, S, S_kv = dots.shape

        causal_mask = torch.triu(torch.ones(S, S_kv, device=dots.device), diagonal=1).bool()
        dots = dots.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            dots = dots.masked_fill(padding_mask, float('-inf'))

        return dots
        

def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    return Transformer(
        vocab_size=vocab_size,
        d=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1
    )

