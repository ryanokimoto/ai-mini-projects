import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        tagset_size,
        num_layers=2,
        dropout=0.5,
        pad_idx=0
    ):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, sentences, lengths):
        batch_size, max_len = sentences.size()
        
        embeds = self.embedding(sentences)
        embeds = self.dropout(embeds)
        
        packed_embeds = pack_padded_sequence(
            embeds, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        
        packed_lstm_out, _ = self.lstm(packed_embeds)
        
        lstm_out, _ = pad_packed_sequence(
            packed_lstm_out, 
            batch_first=True, 
            total_length=max_len
        )

        lstm_out = self.dropout(lstm_out)
        
        tag_scores = self.hidden2tag(lstm_out)
        
        return tag_scores
