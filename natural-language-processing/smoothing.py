"""
N-gram Language Model with Linear Interpolation Smoothing
Assignment 2 - Part 2: Smoothing with Linear Interpolation
NLP 201: Natural Language Processing I

This module implements linear interpolation smoothing between
unigram, bigram, and trigram language models.
"""

import math
from collections import defaultdict, Counter
import sys
import os


class NgramLanguageModel:
    """
    N-gram Language Model with MLE (no smoothing)
    
    This is the base model used for interpolation.
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
        self.token_to_unk = {}
        
    def read_file(self, filename):
        """Read sentences from a file."""
        if not os.path.exists(filename):
            print(f"ERROR: File '{filename}' not found!")
            sys.exit(1)
            
        sentences = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split()
                    if tokens:
                        sentences.append(tokens)
        
        print(f"Read {len(sentences)} sentences from {filename}")
        return sentences
    
    def build_vocabulary(self, sentences):
        """Build vocabulary by identifying tokens that appear >= unk_threshold times."""
        token_counts = Counter()
        for sentence in sentences:
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
        
        print(f"Vocabulary size (including <UNK> and <STOP>, excluding <START>): {len(self.vocab)}")
    
    def preprocess_sentence(self, sentence, is_training=True):
        """Preprocess a sentence by adding special tokens and handling <UNK>."""
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
        """Extract n-grams from a preprocessed sentence."""
        ngrams = []
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, sentences):
        """Train the n-gram language model."""
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
    
    def get_probability(self, ngram):
        """Calculate the MLE probability of an n-gram."""
        if self.n == 1:
            total_count = sum(self.ngram_counts.values())
            return self.ngram_counts[ngram] / total_count if total_count > 0 else 0
        else:
            context = ngram[:-1]
            context_count = self.context_counts[context]
            if context_count == 0:
                return 0
            return self.ngram_counts[ngram] / context_count


class InterpolatedLanguageModel:
    """
    Interpolated Language Model using linear interpolation of unigram, bigram, and trigram models.
    
    Formula: P'(w_j | w_{j-2}, w_{j-1}) = λ1*P(w_j) + λ2*P(w_j|w_{j-1}) + λ3*P(w_j|w_{j-2},w_{j-1})
    where λ1 + λ2 + λ3 = 1
    """
    
    def __init__(self, unigram_model, bigram_model, trigram_model, lambda1, lambda2, lambda3):
        """
        Initialize the interpolated language model.
        
        Args:
            unigram_model: Trained unigram model
            bigram_model: Trained bigram model
            trigram_model: Trained trigram model
            lambda1: Weight for unigram model
            lambda2: Weight for bigram model
            lambda3: Weight for trigram model
        """
        self.unigram_model = unigram_model
        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        # Verify lambdas sum to 1
        if not math.isclose(lambda1 + lambda2 + lambda3, 1.0):
            raise ValueError(f"Lambdas must sum to 1: {lambda1} + {lambda2} + {lambda3} = {lambda1+lambda2+lambda3}")
    
    def get_interpolated_probability(self, word, context):
        """
        Calculate the interpolated probability for a word given context.
        
        Args:
            word: The word to calculate probability for
            context: Tuple of context words (length 0, 1, or 2)
            
        Returns:
            Interpolated probability
        """
        # Unigram probability P(word)
        unigram = (word,)
        p_unigram = self.unigram_model.get_probability(unigram)
        
        # Bigram probability P(word | context[-1])
        if len(context) >= 1:
            bigram = (context[-1], word)
            p_bigram = self.bigram_model.get_probability(bigram)
        else:
            p_bigram = 0
        
        # Trigram probability P(word | context[-2], context[-1])
        if len(context) >= 2:
            trigram = (context[-2], context[-1], word)
            p_trigram = self.trigram_model.get_probability(trigram)
        else:
            p_trigram = 0
        
        # Linear interpolation
        interpolated_prob = (self.lambda1 * p_unigram + 
                            self.lambda2 * p_bigram + 
                            self.lambda3 * p_trigram)
        
        return interpolated_prob
    
    def sentence_log_probability(self, sentence):
        """
        Calculate the log probability of a sentence using interpolation.
        
        Args:
            sentence: Preprocessed sentence with <START> and <STOP> tokens
            
        Returns:
            Log probability of the sentence
        """
        log_prob = 0.0
        
        # Process the sentence token by token
        # For trigram, we need 2 <START> tokens in the context
        for i in range(2, len(sentence)):  # Start from position 2 (after two <START> tokens)
            word = sentence[i]
            context = (sentence[i-2], sentence[i-1])
            
            prob = self.get_interpolated_probability(word, context)
            
            # Handle zero probabilities
            if prob == 0:
                prob = 1e-10
            
            log_prob += math.log(prob)
        
        return log_prob
    
    def calculate_perplexity(self, sentences, show_progress=False):
        """
        Calculate perplexity on a set of sentences using interpolation.
        
        Args:
            sentences: List of sentences
            show_progress: Whether to show progress indicators
            
        Returns:
            Perplexity value
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        if show_progress:
            print(f"Processing {len(sentences)} sentences...")
        
        for idx, sentence in enumerate(sentences):
            if show_progress and (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(sentences)} sentences...")
            
            # Preprocess sentence (adds 2 <START> and 1 <STOP>)
            processed_sentence = self.trigram_model.preprocess_sentence(sentence, is_training=False)
            
            # M = tokens including <STOP> but not <START>
            m = len(processed_sentence) - 2  # Subtract 2 <START> tokens
            total_tokens += m
            
            # Calculate log probability
            log_prob = self.sentence_log_probability(processed_sentence)
            total_log_prob += log_prob
        
        if show_progress:
            print(f"  Done! Processed {len(sentences)} sentences.")
        
        # Calculate perplexity
        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity


def test_interpolation_debug():
    """Test interpolation with debug example 'HDTV .'"""
    print("\n" + "=" * 80)
    print("DEBUG TEST FOR INTERPOLATION: 'HDTV .'")
    print("=" * 80)
    
    # Create debug file
    debug_file = 'debug_hdtv.txt'
    with open(debug_file, 'w') as f:
        f.write('HDTV .\n')
    
    # Train models
    unigram = NgramLanguageModel(n=1, unk_threshold=3)
    bigram = NgramLanguageModel(n=2, unk_threshold=3)
    trigram = NgramLanguageModel(n=3, unk_threshold=3)
    
    # Need to train on actual data first
    train_file = 'A2-Data/1b_benchmark.train.tokens'
    train_sentences = unigram.read_file(train_file)
    
    print("\nTraining models for debug test...")
    unigram.train(train_sentences)
    
    bigram.vocab = unigram.vocab
    bigram.token_to_unk = unigram.token_to_unk
    bigram.train(train_sentences)
    
    trigram.vocab = unigram.vocab
    trigram.token_to_unk = unigram.token_to_unk
    trigram.train(train_sentences)
    
    # Test with λ1=0.1, λ2=0.3, λ3=0.6
    debug_sentences = [['HDTV', '.']]
    interpolated = InterpolatedLanguageModel(unigram, bigram, trigram, 0.1, 0.3, 0.6)
    
    perplexity = interpolated.calculate_perplexity(debug_sentences)
    print(f"\nλ1=0.1, λ2=0.3, λ3=0.6")
    print(f"Perplexity: {perplexity:.1f} (expected: 48.1)")
    
    # Clean up
    if os.path.exists(debug_file):
        os.remove(debug_file)
    
    return unigram, bigram, trigram


def experiment_lambda_values(unigram, bigram, trigram, train_sentences, dev_sentences):
    """
    Experiment with different lambda values.
    
    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENTING WITH DIFFERENT LAMBDA VALUES")
    print("=" * 80)
    
    # Different lambda combinations to try
    lambda_sets = [
        (0.3, 0.3, 0.4),   # Required by assignment
        (0.1, 0.3, 0.6),   # Higher weight on trigram
        (0.2, 0.3, 0.5),   # Balanced
        (0.1, 0.4, 0.5),   # Higher weight on bigram and trigram
        (0.15, 0.35, 0.5), # Another balanced option
    ]
    
    results = []
    
    for lambda1, lambda2, lambda3 in lambda_sets:
        print(f"\nTesting λ1={lambda1}, λ2={lambda2}, λ3={lambda3}")
        
        model = InterpolatedLanguageModel(unigram, bigram, trigram, lambda1, lambda2, lambda3)
        
        train_ppl = model.calculate_perplexity(train_sentences, show_progress=True)
        dev_ppl = model.calculate_perplexity(dev_sentences, show_progress=True)
        
        print(f"  Training perplexity: {train_ppl:.2f}")
        print(f"  Development perplexity: {dev_ppl:.2f}")
        
        results.append({
            'lambda1': lambda1,
            'lambda2': lambda2,
            'lambda3': lambda3,
            'train_ppl': train_ppl,
            'dev_ppl': dev_ppl
        })
    
    return results


def experiment_half_training_data(train_sentences, dev_sentences, unk_threshold=3):
    """
    Experiment: Using half of the training data.
    Question 3: Would perplexity increase or decrease on unseen data?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT: HALF TRAINING DATA")
    print("=" * 80)
    
    # Use only half of the training data
    half_size = len(train_sentences) // 2
    half_train = train_sentences[:half_size]
    
    print(f"\nTraining on {len(half_train)} sentences (half of {len(train_sentences)})")
    
    # Train new models
    unigram_half = NgramLanguageModel(n=1, unk_threshold=unk_threshold)
    unigram_half.train(half_train)
    
    bigram_half = NgramLanguageModel(n=2, unk_threshold=unk_threshold)
    bigram_half.vocab = unigram_half.vocab
    bigram_half.token_to_unk = unigram_half.token_to_unk
    bigram_half.train(half_train)
    
    trigram_half = NgramLanguageModel(n=3, unk_threshold=unk_threshold)
    trigram_half.vocab = unigram_half.vocab
    trigram_half.token_to_unk = unigram_half.token_to_unk
    trigram_half.train(half_train)
    
    # Use a good lambda combination
    model_half = InterpolatedLanguageModel(unigram_half, bigram_half, trigram_half, 0.1, 0.3, 0.6)
    
    dev_ppl_half = model_half.calculate_perplexity(dev_sentences, show_progress=True)
    
    print(f"\nDevelopment perplexity with HALF training data: {dev_ppl_half:.2f}")
    
    return dev_ppl_half


def experiment_unk_threshold(train_sentences, dev_sentences):
    """
    Experiment: Different UNK thresholds.
    Question 4: Converting tokens < 5 times vs only fraction appearing once.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT: DIFFERENT UNK THRESHOLDS")
    print("=" * 80)
    
    results = []
    
    for unk_threshold in [3, 5]:  # 3 is default, 5 is the comparison
        print(f"\n--- UNK Threshold = {unk_threshold} ---")
        
        unigram_unk = NgramLanguageModel(n=1, unk_threshold=unk_threshold)
        unigram_unk.train(train_sentences)
        
        bigram_unk = NgramLanguageModel(n=2, unk_threshold=unk_threshold)
        bigram_unk.vocab = unigram_unk.vocab
        bigram_unk.token_to_unk = unigram_unk.token_to_unk
        bigram_unk.train(train_sentences)
        
        trigram_unk = NgramLanguageModel(n=3, unk_threshold=unk_threshold)
        trigram_unk.vocab = unigram_unk.vocab
        trigram_unk.token_to_unk = unigram_unk.token_to_unk
        trigram_unk.train(train_sentences)
        
        model_unk = InterpolatedLanguageModel(unigram_unk, bigram_unk, trigram_unk, 0.1, 0.3, 0.6)
        
        dev_ppl_unk = model_unk.calculate_perplexity(dev_sentences, show_progress=True)
        
        print(f"Development perplexity with threshold {unk_threshold}: {dev_ppl_unk:.2f}")
        
        results.append({
            'unk_threshold': unk_threshold,
            'vocab_size': len(unigram_unk.vocab),
            'dev_ppl': dev_ppl_unk
        })
    
    return results


def main():
    """Main function for Part 2: Linear Interpolation Smoothing"""
    
    print("=" * 80)
    print("PART 2: LINEAR INTERPOLATION SMOOTHING")
    print("=" * 80)
    
    # File paths
    train_file = 'A2-Data/1b_benchmark.train.tokens'
    dev_file = 'A2-Data/1b_benchmark.dev.tokens'
    test_file = 'A2-Data/1b_benchmark.test.tokens'
    
    # Check files exist
    for filepath in [train_file, dev_file, test_file]:
        if not os.path.exists(filepath):
            print(f"ERROR: File '{filepath}' not found!")
            sys.exit(1)
    
    # Step 1: Debug test
    unigram, bigram, trigram = test_interpolation_debug()
    
    # Read data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train_sentences = unigram.read_file(train_file)
    dev_sentences = unigram.read_file(dev_file)
    test_sentences = unigram.read_file(test_file)
    
    # Step 2: Experiment with lambda values (Deliverable 1)
    lambda_results = experiment_lambda_values(unigram, bigram, trigram, 
                                             train_sentences, dev_sentences)
    
    # Step 3: Find best lambdas and evaluate on test set (Deliverable 2)
    print("\n" + "=" * 80)
    print("SELECTING BEST HYPERPARAMETERS AND EVALUATING ON TEST SET")
    print("=" * 80)
    
    best_result = min(lambda_results, key=lambda x: x['dev_ppl'])
    best_lambda1 = best_result['lambda1']
    best_lambda2 = best_result['lambda2']
    best_lambda3 = best_result['lambda3']
    
    print(f"\nBest hyperparameters (lowest dev perplexity):")
    print(f"  λ1={best_lambda1}, λ2={best_lambda2}, λ3={best_lambda3}")
    print(f"  Development perplexity: {best_result['dev_ppl']:.2f}")
    
    best_model = InterpolatedLanguageModel(unigram, bigram, trigram, 
                                          best_lambda1, best_lambda2, best_lambda3)
    test_ppl = best_model.calculate_perplexity(test_sentences, show_progress=True)
    
    print(f"\nTest set perplexity with best hyperparameters: {test_ppl:.2f}")
    
    # Step 4: Half training data experiment (Deliverable 3)
    dev_ppl_full = best_result['dev_ppl']
    dev_ppl_half = experiment_half_training_data(train_sentences, dev_sentences)
    
    # Step 5: UNK threshold experiment (Deliverable 4)
    unk_results = experiment_unk_threshold(train_sentences, dev_sentences)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    print("\n1. Lambda Values Experiments:")
    print(f"{'λ1':<8} {'λ2':<8} {'λ3':<8} {'Train PPL':<12} {'Dev PPL':<12}")
    print("-" * 56)
    for result in lambda_results:
        print(f"{result['lambda1']:<8.2f} {result['lambda2']:<8.2f} {result['lambda3']:<8.2f} "
              f"{result['train_ppl']:<12.2f} {result['dev_ppl']:<12.2f}")
    
    print(f"\n2. Best Model on Test Set:")
    print(f"   Hyperparameters: λ1={best_lambda1}, λ2={best_lambda2}, λ3={best_lambda3}")
    print(f"   Test perplexity: {test_ppl:.2f}")
    
    print(f"\n3. Half Training Data Experiment:")
    print(f"   Full training data dev perplexity: {dev_ppl_full:.2f}")
    print(f"   Half training data dev perplexity: {dev_ppl_half:.2f}")
    print(f"   Change: {'+' if dev_ppl_half > dev_ppl_full else ''}{dev_ppl_half - dev_ppl_full:.2f}")
    print(f"   Result: Perplexity {'INCREASED' if dev_ppl_half > dev_ppl_full else 'DECREASED'}")
    
    print(f"\n4. UNK Threshold Experiment:")
    print(f"{'Threshold':<12} {'Vocab Size':<12} {'Dev PPL':<12}")
    print("-" * 36)
    for result in unk_results:
        print(f"{result['unk_threshold']:<12} {result['vocab_size']:<12} {result['dev_ppl']:<12.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()