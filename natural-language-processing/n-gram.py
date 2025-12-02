"""
N-gram Language Model Implementation
Assignment 2 - Part 1: N-gram Language Modeling
NLP 201: Natural Language Processing I

This module implements unigram, bigram, and trigram language models
using Maximum Likelihood Estimation (MLE) without smoothing.
"""

import math
from collections import defaultdict, Counter
import sys
import os


class NgramLanguageModel:
    """
    N-gram Language Model with MLE (no smoothing)
    
    Attributes:
        n: The n-gram size (1 for unigram, 2 for bigram, 3 for trigram)
        vocab: Set of unique tokens in the vocabulary
        ngram_counts: Dictionary storing counts of n-grams
        context_counts: Dictionary storing counts of (n-1)-grams (contexts)
        unk_threshold: Minimum frequency for a token to be in vocabulary (default: 3)
    """
    
    def __init__(self, n, unk_threshold=3):
        """
        Initialize the n-gram language model.
        
        Args:
            n: N-gram size (1, 2, or 3)
            unk_threshold: Tokens appearing less than this many times become <UNK>
        """
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.token_to_unk = {}  # Maps tokens to either themselves or <UNK>
        
    def read_file(self, filename):
        """
        Read sentences from a file.
        
        Args:
            filename: Path to the input file
            
        Returns:
            List of sentences, where each sentence is a list of tokens
        """
        if not os.path.exists(filename):
            print(f"ERROR: File '{filename}' not found!")
            print(f"Please make sure the data files are in the same directory as this script.")
            sys.exit(1)
            
        sentences = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    tokens = line.split()
                    if tokens:
                        sentences.append(tokens)
        
        print(f"Read {len(sentences)} sentences from {filename}")
        return sentences
    
    def build_vocabulary(self, sentences):
        """
        Build vocabulary by identifying tokens that appear >= unk_threshold times.
        Tokens that appear less frequently are mapped to <UNK>.
        
        Args:
            sentences: List of sentences (each sentence is a list of tokens)
        """
        # Count all tokens
        token_counts = Counter()
        for sentence in sentences:
            if not isinstance(sentence, list):
                print(f"ERROR: Expected sentence to be a list, got {type(sentence)}")
                print(f"First few characters: {str(sentence)[:100]}")
                sys.exit(1)
            for token in sentence:
                token_counts[token] += 1
        
        # Build vocabulary with tokens that appear >= unk_threshold times
        self.vocab = {'<UNK>', '<STOP>'}  # Special tokens always in vocab
        self.token_to_unk = {}
        
        for token, count in token_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(token)
                self.token_to_unk[token] = token
            else:
                self.token_to_unk[token] = '<UNK>'
        
        print(f"Vocabulary size (including <UNK> and <STOP>, excluding <START>): {len(self.vocab)}")
    
    def preprocess_sentence(self, sentence, is_training=True):
        """
        Preprocess a sentence by adding special tokens and handling <UNK>.
        
        Args:
            sentence: List of tokens
            is_training: If False, map unseen tokens to <UNK>
            
        Returns:
            Preprocessed list of tokens with <START> and <STOP> tokens
        """
        # Map tokens to <UNK> if necessary
        processed = []
        for token in sentence:
            if is_training:
                processed.append(self.token_to_unk.get(token, token))
            else:
                # During testing, unseen words become <UNK>
                if token in self.vocab:
                    processed.append(token)
                elif token in self.token_to_unk:
                    processed.append(self.token_to_unk[token])
                else:
                    processed.append('<UNK>')
        
        # Add <START> tokens (n-1 of them) and one <STOP> token
        start_tokens = ['<START>'] * (self.n - 1)
        return start_tokens + processed + ['<STOP>']
    
    def extract_ngrams(self, sentence):
        """
        Extract n-grams from a preprocessed sentence.
        
        Args:
            sentence: Preprocessed sentence with <START> and <STOP> tokens
            
        Returns:
            List of n-grams (each n-gram is a tuple)
        """
        ngrams = []
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, sentences):
        """
        Train the n-gram language model on the given sentences.
        
        Args:
            sentences: List of sentences (each sentence is a list of tokens)
        """
        print(f"Training {self.n}-gram model...")
        
        # First pass: build vocabulary
        self.build_vocabulary(sentences)
        
        # Second pass: count n-grams
        for sentence in sentences:
            processed_sentence = self.preprocess_sentence(sentence, is_training=True)
            ngrams = self.extract_ngrams(processed_sentence)
            
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                
                # For n > 1, also count the context (first n-1 words)
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        
        print(f"Total unique {self.n}-grams: {len(self.ngram_counts)}")
        if self.n > 1:
            print(f"Total unique {self.n-1}-grams (contexts): {len(self.context_counts)}")
    
    def get_probability(self, ngram):
        """
        Calculate the MLE probability of an n-gram.
        
        Args:
            ngram: Tuple of n tokens
            
        Returns:
            Probability of the n-gram (float)
        """
        if self.n == 1:
            # Unigram: P(w) = count(w) / total_tokens
            total_count = sum(self.ngram_counts.values())
            return self.ngram_counts[ngram] / total_count if total_count > 0 else 0
        else:
            # N-gram: P(w_n | w_1...w_{n-1}) = count(w_1...w_n) / count(w_1...w_{n-1})
            context = ngram[:-1]
            context_count = self.context_counts[context]
            if context_count == 0:
                return 0
            return self.ngram_counts[ngram] / context_count
    
    def sentence_log_probability(self, sentence, bigram_model=None):
        """
        Calculate the log probability of a sentence.
        
        Args:
            sentence: Preprocessed sentence with <START> and <STOP> tokens
            bigram_model: For trigram model, use bigram for first word after <START>
            
        Returns:
            Log probability of the sentence
        """
        log_prob = 0.0
        ngrams = self.extract_ngrams(sentence)
        
        for i, ngram in enumerate(ngrams):
            # Special case for trigram: use bigram probability for first word after <START>
            if self.n == 3 and i == 0 and bigram_model is not None:
                # First trigram is (<START>, <START>, first_word)
                # Use bigram probability P(first_word | <START>)
                bigram = ('<START>', ngram[2])
                prob = bigram_model.get_probability(bigram)
            else:
                prob = self.get_probability(ngram)
            
            # Handle zero probabilities
            if prob == 0:
                # Use a small probability to avoid log(0)
                prob = 1e-10
            
            log_prob += math.log(prob)
        
        return log_prob
    
    def calculate_perplexity(self, sentences, bigram_model=None):
        """
        Calculate perplexity on a set of sentences.
        
        Perplexity = exp(-1/M * sum(log P(sentence)))
        where M = total number of tokens including <STOP> but not <START>
        
        Args:
            sentences: List of sentences
            bigram_model: For trigram model, use bigram for first word after <START>
            
        Returns:
            Perplexity value (float)
        """
        total_log_prob = 0.0
        total_tokens = 0  # M in the formula
        
        for sentence in sentences:
            processed_sentence = self.preprocess_sentence(sentence, is_training=False)
            
            # Calculate M: tokens including <STOP> but not <START>
            # M = len(original_tokens) + 1 (<STOP>)
            m = len(processed_sentence) - (self.n - 1)  # Subtract <START> tokens
            total_tokens += m
            
            # Calculate log probability
            log_prob = self.sentence_log_probability(processed_sentence, bigram_model)
            total_log_prob += log_prob
        
        # Calculate perplexity
        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity


def main():
    """
    Main function to train and evaluate n-gram language models.
    """
    # File paths - check if files exist
    train_file = 'A2-Data/1b_benchmark.train.tokens'
    dev_file = 'A2-Data/1b_benchmark.dev.tokens'
    test_file = 'A2-Data/1b_benchmark.test.tokens'
    
    # Check if files exist
    for filepath in [train_file, dev_file, test_file]:
        if not os.path.exists(filepath):
            print(f"ERROR: Required file '{filepath}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            print("\nPlease ensure all three data files are in the same directory as this script:")
            print(f"  - {train_file}")
            print(f"  - {dev_file}")
            print(f"  - {test_file}")
            sys.exit(1)
    
    print("=" * 80)
    print("N-gram Language Model Training and Evaluation")
    print("=" * 80)
    
    # Train unigram model
    print("\n" + "=" * 80)
    print("UNIGRAM MODEL")
    print("=" * 80)
    unigram_model = NgramLanguageModel(n=1, unk_threshold=3)
    train_sentences = unigram_model.read_file(train_file)
    unigram_model.train(train_sentences)
    
    # Train bigram model
    print("\n" + "=" * 80)
    print("BIGRAM MODEL")
    print("=" * 80)
    bigram_model = NgramLanguageModel(n=2, unk_threshold=3)
    bigram_model.vocab = unigram_model.vocab  # Use same vocabulary
    bigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences_bi = bigram_model.read_file(train_file)
    bigram_model.train(train_sentences_bi)
    
    # Train trigram model
    print("\n" + "=" * 80)
    print("TRIGRAM MODEL")
    print("=" * 80)
    trigram_model = NgramLanguageModel(n=3, unk_threshold=3)
    trigram_model.vocab = unigram_model.vocab  # Use same vocabulary
    trigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences_tri = trigram_model.read_file(train_file)
    trigram_model.train(train_sentences_tri)
    
    # Debug test with "HDTV ."
    print("\n" + "=" * 80)
    print("DEBUG TEST: 'HDTV .'")
    print("=" * 80)
    debug_sentences = [['HDTV', '.']]
    
    uni_debug_ppl = unigram_model.calculate_perplexity(debug_sentences)
    bi_debug_ppl = bigram_model.calculate_perplexity(debug_sentences)
    tri_debug_ppl = trigram_model.calculate_perplexity(debug_sentences, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_debug_ppl:.1f} (expected: 658)")
    print(f"Bigram perplexity: {bi_debug_ppl:.1f} (expected: 63.7)")
    print(f"Trigram perplexity: {tri_debug_ppl:.1f} (expected: 39.5)")
    
    # Evaluate on training set
    print("\n" + "=" * 80)
    print("PERPLEXITY ON TRAINING SET")
    print("=" * 80)
    
    uni_train_ppl = unigram_model.calculate_perplexity(train_sentences)
    bi_train_ppl = bigram_model.calculate_perplexity(train_sentences_bi)
    tri_train_ppl = trigram_model.calculate_perplexity(train_sentences_tri, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_train_ppl:.2f}")
    print(f"Bigram perplexity: {bi_train_ppl:.2f}")
    print(f"Trigram perplexity: {tri_train_ppl:.2f}")
    
    # Evaluate on development set
    print("\n" + "=" * 80)
    print("PERPLEXITY ON DEVELOPMENT SET")
    print("=" * 80)
    dev_sentences = unigram_model.read_file(dev_file)
    
    uni_dev_ppl = unigram_model.calculate_perplexity(dev_sentences)
    bi_dev_ppl = bigram_model.calculate_perplexity(dev_sentences)
    tri_dev_ppl = trigram_model.calculate_perplexity(dev_sentences, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_dev_ppl:.2f}")
    print(f"Bigram perplexity: {bi_dev_ppl:.2f}")
    print(f"Trigram perplexity: {tri_dev_ppl:.2f}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("PERPLEXITY ON TEST SET")
    print("=" * 80)
    test_sentences = unigram_model.read_file(test_file)
    
    uni_test_ppl = unigram_model.calculate_perplexity(test_sentences)
    bi_test_ppl = bigram_model.calculate_perplexity(test_sentences)
    tri_test_ppl = trigram_model.calculate_perplexity(test_sentences, bigram_model=bigram_model)
    
    print(f"Unigram perplexity: {uni_test_ppl:.2f}")
    print(f"Bigram perplexity: {bi_test_ppl:.2f}")
    print(f"Trigram perplexity: {tri_test_ppl:.2f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<15} {'Training':<15} {'Development':<15} {'Test':<15}")
    print("-" * 60)
    print(f"{'Unigram':<15} {uni_train_ppl:<15.2f} {uni_dev_ppl:<15.2f} {uni_test_ppl:<15.2f}")
    print(f"{'Bigram':<15} {bi_train_ppl:<15.2f} {bi_dev_ppl:<15.2f} {bi_test_ppl:<15.2f}")
    print(f"{'Trigram':<15} {tri_train_ppl:<15.2f} {tri_dev_ppl:<15.2f} {tri_test_ppl:<15.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()