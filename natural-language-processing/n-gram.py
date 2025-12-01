import sys

class LanguageModel:
    
    def __init__(self, n, unk_threshold=3):
        self.n = n
        self.unk_threshold = 3

        self.vocab = set()
        self.ngrams = {}
        self.context_counts = {}
        self.tokens_to_unk = {}

    def read_file(self, filename):
        sentences = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
        return sentences
    
    def build_vocabulary(self, sentences):
        token_counts = {}
        for sentence in sentences:
            token_counts.update(sentence)

        self.vocab = {'UNK', 'STOP'}
        self.token_to_unk = {}

        for token, count in token_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(token)
                self.token_to_unk[token] = token
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def preprocess_sentences(self, sentence, is_training=True):
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
                    processed.append('UNK')
        start_tokens = ['<START>'] * (self.n - 1)
        return start_tokens + processed + ['<STOP>']

    def build_ngrams(self, sentence):
        ngrams = []
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams

    def train(self, sentences):
        print("Training ...")
        self.build_vocabulary(sentences)
        for sentence in sentences:
            processed_sentence = self.preprocess_sentences(sentence, is_training=True)
            ngrams = self.build_ngrams(processed_sentence)

            for ngram in ngrams:
                self.ngrams[ngram] += 1
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1
        print(f"Total unique {self.n}-grams: {len(self.ngram_counts)}")
        if self.n > 1:
            print(f"Total unique {self.n-1}-grams (contexts): {len(self.context_counts)}")

    def get_probability(self, ngram):
        if self.n == 1:
            total_count = sum(self.ngrams.values())
            return self.ngrams.get(ngram, 0) / total_count if total_count > 0 else 0
        else:
            context = ngram[:-1]
            context_count = self.context_counts[context]
            if context_count == 0:
                return 0
            return self.ngrams.get(ngram, 0) / context_count
    
    def sentence_log_probability(self, sentence):
        log_prob = 0.0
        ngrams = self.build_ngrams(sentence)

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
            processed_sentence = self.preprocess_sentences(sentence, is_training=False)
            m = len(processed_sentence) - (self.n - 1)
            total_tokens += m

            log_prob = self.sentence_log_probability(processed_sentence, bigram_model)
            total_log_prob += log_prob

        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity

def main():
    """
    Main function to train and evaluate n-gram language models.
    """
    # File paths
    train_file = 'A2-Data/1b_benchmark.train.tokens'
    dev_file = 'A2-Data/1b_benchmark.dev.tokens'
    test_file = 'A2-Data/1b_benchmark.test.tokens'
    
    print("=" * 80)
    print("N-gram Language Model Training and Evaluation")
    print("=" * 80)
    
    # Train unigram model
    print("\n" + "=" * 80)
    print("UNIGRAM MODEL")
    print("=" * 80)
    unigram_model = LanguageModel(n=1, unk_threshold=3)
    train_sentences = unigram_model.read_file(train_file)
    unigram_model.train(train_sentences)
    
    # Train bigram model
    print("\n" + "=" * 80)
    print("BIGRAM MODEL")
    print("=" * 80)
    bigram_model = LanguageModel(n=2, unk_threshold=3)
    bigram_model.vocab = unigram_model.vocab  # Use same vocabulary
    bigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences = bigram_model.read_file(train_file)
    bigram_model.train(train_sentences)
    
    # Train trigram model
    print("\n" + "=" * 80)
    print("TRIGRAM MODEL")
    print("=" * 80)
    trigram_model = LanguageModel(n=3, unk_threshold=3)
    trigram_model.vocab = unigram_model.vocab  # Use same vocabulary
    trigram_model.token_to_unk = unigram_model.token_to_unk
    train_sentences = trigram_model.read_file(train_file)
    trigram_model.train(train_sentences)
    
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
    train_sentences = unigram_model.read_file(train_file)
    
    uni_train_ppl = unigram_model.calculate_perplexity(train_sentences)
    bi_train_ppl = bigram_model.calculate_perplexity(train_sentences)
    tri_train_ppl = trigram_model.calculate_perplexity(train_sentences, bigram_model=bigram_model)
    
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