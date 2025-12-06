import csv
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def build_vocab(train_path):
    words = set()
    tags = set()
    
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row['utterances'].strip().lower().split()
            labels = row['IOB Slot tags'].strip().split()
            
            words.update(sentence)
            tags.update(labels)
    
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word in sorted(words):
        word2idx[word] = len(word2idx)
    
    tag2idx = {'<PAD>': 0}
    for tag in sorted(tags):
        tag2idx[tag] = len(tag2idx)
    
    return word2idx, tag2idx


class SlotTaggingDataset(Dataset):
    def __init__(self, filepath, word2idx, tag2idx, is_test=False):
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.is_test = is_test
        self.samples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            print(f"CSV columns: {columns}")
            
            for row in reader:
                text_col = None
                for col in columns:
                    col_lower = col.lower()
                    if col_lower != 'id' and 'tag' not in col_lower and 'slot' not in col_lower:
                        text_col = col
                        break
                
                if text_col is None:
                    text_col = columns[1] if len(columns) > 1 else columns[0]
                
                sentence = row[text_col].strip().lower().split()
                
                if is_test:
                    tags = None
                else:
                    tag_col = None
                    for col in columns:
                        col_lower = col.lower()
                        if 'tag' in col_lower or 'slot' in col_lower:
                            tag_col = col
                            break
                    
                    if tag_col is None:
                        tag_col = columns[-1] 
                    
                    tags = row[tag_col].strip().split()
                
                self.samples.append((sentence, tags))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sentence, tags = self.samples[idx]
        
        word_indices = []
        for word in sentence:
            if word in self.word2idx:
                word_indices.append(self.word2idx[word])
            else:
                word_indices.append(self.word2idx['<UNK>'])
        
        word_tensor = torch.tensor(word_indices, dtype=torch.long)
        
        if self.is_test:
            return word_tensor, None, len(sentence), idx
        else:
            tag_indices = [self.tag2idx[tag] for tag in tags]
            tag_tensor = torch.tensor(tag_indices, dtype=torch.long)
            return word_tensor, tag_tensor, len(sentence), idx


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    
    words = [item[0] for item in batch]
    tags = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    global_indices = [item[3] for item in batch]
    
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    
    if tags[0] is not None:
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    else:
        tags_padded = None
    
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return words_padded, tags_padded, lengths, global_indices
