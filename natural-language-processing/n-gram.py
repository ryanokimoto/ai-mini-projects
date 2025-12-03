import math
from collections import defaultdict, Counter
import sys
import os


class NgramLanguageModel:    
    def __init__(self, n, unk_threshold=3):
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.token_to_unk = {}
        
    def read_file(self, filename):
        sentences = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    tokens = line.split()
                    if tokens:
                        sentences.append(tokens)
        return sentences
    
    def build_vocabulary(self, sentences):
        token_counts = Counter()
        for sentence in sentences:
            if not isinstance(sentence, list):
                sys.exit("Error: Each sentence should be a list of tokens.")
            for token in sentence:
                token_counts[token] += 1
        
        self.vocab = {'<UNK>', '<STOP>'}
        self.token_to_unk = {}
        
        for token, count in token_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(token)
                self.token_to_unk[token] = token
            else:
                self.token_to_unk[token] = '<UNK>'
        
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def preprocess_sentence(self, sentence, is_training=True):
        processed = []
        for token in sentence:
            if is_training:
                processed.append(self.token_to_unk.get(token, token))
            else:
                if token in self.vocab:
                    processed.append(token)
                elif token in self.token_to_unk:
                    processed.append(self.token_to_unk[token])
                else:
                    processed.append('<UNK>')
        
        start_tokens = ['<START>'] * (self.n - 1)
        return start_tokens + processed + ['<STOP>']
    
    def extract_ngrams(self, sentence):
        ngrams = []
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, sentences):
        print(f"Training {self.n}-gram model...")
        self.build_vocabulary(sentences)
        
        for sentence in sentences:
            processed_sentence = self.preprocess_sentence(sentence, is_training=True)
            ngrams = self.extract_ngrams(processed_sentence)
            
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        
        print(f"Total unique {self.n}-grams: {len(self.ngram_counts)}")
        if self.n > 1:
            print(f"Total unique {self.n-1}-grams (contexts): {len(self.context_counts)}")
    
    def get_probability(self, ngram):
        if self.n == 1:
            total_count = sum(self.ngram_counts.values())
            return self.ngram_counts[ngram] / total_count if total_count > 0 else 0
        else:
            context = ngram[:-1]
            context_count = self.context_counts[context]
            if context_count == 0:
                return 0
            return self.ngram_counts[ngram] / context_count
    
    def sentence_log_probability(self, sentence, bigram_model=None):
        log_prob = 0.0
        ngrams = self.extract_ngrams(sentence)
        
        for i, ngram in enumerate(ngrams):
            if self.n == 3 and i == 0 and bigram_model is not None:
                bigram = ('<START>', ngram[2])
                prob = bigram_model.get_probability(bigram)
            else:
                prob = self.get_probability(ngram)
            
            if prob == 0:
                prob = 1e-10
            
            log_prob += math.log(prob)
        
        return log_prob
    
    def calculate_perplexity(self, sentences, bigram_model=None):
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            processed_sentence = self.preprocess_sentence(sentence, is_training=False)
            
            m = len(processed_sentence) - (self.n - 1)
            total_tokens += m

            log_prob = self.sentence_log_probability(processed_sentence, bigram_model)
            total_log_prob += log_prob
        
        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity


def main():
    train_file = 'A2-Data/1b_benchmark.train.tokens'
    dev_file = 'A2-Data/1b_benchmark.dev.tokens'
    test_file = 'A2-Data/1b_benchmark.test.tokens'
    
    print("\n")
    print("UNIGRAM MODEL")
    unigram_model = NgramLanguageModel(n=1, unk_threshold=3)
    train_sentences = unigram_model.read_file(train_file)
    unigram_model.train(train_sentences)
    
    print("\n")
    print("BIGRAM MODEL")
    bigram_model = NgramLanguageModel(n=2, unk_threshold=3)
    bigram_model.vocab = unigram_model.vocab
    bigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences_bi = bigram_model.read_file(train_file)
    bigram_model.train(train_sentences_bi)
    
    print("\n")
    print("TRIGRAM MODEL")
    trigram_model = NgramLanguageModel(n=3, unk_threshold=3)
    trigram_model.vocab = unigram_model.vocab
    trigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences_tri = trigram_model.read_file(train_file)
    trigram_model.train(train_sentences_tri)
    
    # debug_sentences = [['HDTV', '.']]
    
    # uni_debug_ppl = unigram_model.calculate_perplexity(debug_sentences)
    # bi_debug_ppl = bigram_model.calculate_perplexity(debug_sentences)
    # tri_debug_ppl = trigram_model.calculate_perplexity(debug_sentences, bigram_model=bigram_model)
    
    # print(f"Unigram perplexity: {uni_debug_ppl:.1f}")
    # print(f"Bigram perplexity: {bi_debug_ppl:.1f}")
    # print(f"Trigram perplexity: {tri_debug_ppl:.1f}")
    
    print("\n")
    print("PERPLEXITY ON TRAINING SET")
    
    uni_train_ppl = unigram_model.calculate_perplexity(train_sentences)
    bi_train_ppl = bigram_model.calculate_perplexity(train_sentences_bi)
    tri_train_ppl = trigram_model.calculate_perplexity(train_sentences_tri, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_train_ppl:.2f}")
    print(f"Bigram perplexity: {bi_train_ppl:.2f}")
    print(f"Trigram perplexity: {tri_train_ppl:.2f}")
    
    print("\n")
    print("PERPLEXITY ON DEVELOPMENT SET")
    dev_sentences = unigram_model.read_file(dev_file)
    
    uni_dev_ppl = unigram_model.calculate_perplexity(dev_sentences)
    bi_dev_ppl = bigram_model.calculate_perplexity(dev_sentences)
    tri_dev_ppl = trigram_model.calculate_perplexity(dev_sentences, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_dev_ppl:.2f}")
    print(f"Bigram perplexity: {bi_dev_ppl:.2f}")
    print(f"Trigram perplexity: {tri_dev_ppl:.2f}")
    
    print("\n")
    print("PERPLEXITY ON TEST SET")
    test_sentences = unigram_model.read_file(test_file)
    
    uni_test_ppl = unigram_model.calculate_perplexity(test_sentences)
    bi_test_ppl = bigram_model.calculate_perplexity(test_sentences)
    tri_test_ppl = trigram_model.calculate_perplexity(test_sentences, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_test_ppl:.2f}")
    print(f"Bigram perplexity: {bi_test_ppl:.2f}")
    print(f"Trigram perplexity: {tri_test_ppl:.2f}")
    
    print("\n")
    print(f"{'Model':<15} {'Training':<15} {'Development':<15} {'Test':<15}")
    print("-" * 60)
    print(f"{'Unigram':<15} {uni_train_ppl:<15.2f} {uni_dev_ppl:<15.2f} {uni_test_ppl:<15.2f}")
    print(f"{'Bigram':<15} {bi_train_ppl:<15.2f} {bi_dev_ppl:<15.2f} {bi_test_ppl:<15.2f}")
    print(f"{'Trigram':<15} {tri_train_ppl:<15.2f} {tri_dev_ppl:<15.2f} {tri_test_ppl:<15.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()