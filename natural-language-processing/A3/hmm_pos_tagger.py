"""
HMM POS Tagger Implementation for NLP 201 Assignment 3
University of California Santa Cruz

This module implements a Hidden Markov Model (HMM) based Part-of-Speech tagger
with Viterbi decoding and add-alpha smoothing.
"""

import os
import math
import numpy as np
from collections import defaultdict, Counter
import nltk
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


class HMMPOSTagger:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.START_TAG = '<START>'
        self.STOP_TAG = '<STOP>'

        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))
        
        self.vocab = set()
        self.tagset = set()
        
        self.log_transition = {}
        self.log_emission = {}
        
    def train(self, train_sentences):
        print("Training HMM...")
        
        for sentence in train_sentences:
            tags = [self.START_TAG] + [tag for word, tag in sentence] + [self.STOP_TAG]
            words = [self.START_TAG] + [word for word, tag in sentence] + [self.STOP_TAG]
            
            for i in range(len(tags)):
                current_tag = tags[i]
                current_word = words[i]
                
                self.vocab.add(current_word)
                self.tagset.add(current_tag)
                
                self.emission_counts[current_tag][current_word] += 1
                self.tag_counts[current_tag] += 1
                
                if current_word not in [self.START_TAG, self.STOP_TAG]:
                    self.word_tag_counts[current_word][current_tag] += 1
                
                if i > 0:
                    prev_tag = tags[i-1]
                    self.transition_counts[prev_tag][current_tag] += 1
        
        self.decoding_tagset = self.tagset - {self.START_TAG, self.STOP_TAG}
        
        self._compute_log_probabilities()
        
        
    def _compute_log_probabilities(self):
        num_tags = len(self.tagset)
        vocab_size = len(self.vocab)
        
        for prev_tag in self.tagset:
            self.log_transition[prev_tag] = {}
            denominator = self.tag_counts[prev_tag] + self.alpha * num_tags
            
            for curr_tag in self.tagset:
                count = self.transition_counts[prev_tag][curr_tag]
                prob = (count + self.alpha) / denominator
                self.log_transition[prev_tag][curr_tag] = math.log(prob)
        
        for tag in self.tagset:
            self.log_emission[tag] = {}
            denominator = self.tag_counts[tag] + self.alpha * vocab_size
            
            for word in self.vocab:
                count = self.emission_counts[tag][word]
                prob = (count + self.alpha) / denominator
                self.log_emission[tag][word] = math.log(prob)
            
            self.log_emission[tag]['<UNK>'] = math.log(self.alpha / denominator)
    
    def get_transition_log_prob(self, prev_tag, curr_tag):
        return self.log_transition[prev_tag][curr_tag]
    
    def get_emission_log_prob(self, tag, word):
        if word in self.log_emission[tag]:
            return self.log_emission[tag][word]
        return self.log_emission[tag]['<UNK>']
    
    def viterbi_decode(self, words):
        n = len(words)
        if n == 0:
            return []
        
        tags = list(self.decoding_tagset)
        num_tags = len(tags)
        tag_to_idx = {tag: i for i, tag in enumerate(tags)}
        
        pi = np.full((n, num_tags), -np.inf)
        backpointer = np.zeros((n, num_tags), dtype=int)
        
        for t_idx, tag in enumerate(tags):
            trans_prob = self.get_transition_log_prob(self.START_TAG, tag)
            emit_prob = self.get_emission_log_prob(tag, words[0])
            pi[0][t_idx] = trans_prob + emit_prob
        
        for j in range(1, n):
            word = words[j]
            for t_idx, tag in enumerate(tags):
                emit_prob = self.get_emission_log_prob(tag, word)
                
                best_score = -np.inf
                best_prev = 0
                
                for prev_idx, prev_tag in enumerate(tags):
                    trans_prob = self.get_transition_log_prob(prev_tag, tag)
                    score = pi[j-1][prev_idx] + trans_prob
                    
                    if score > best_score:
                        best_score = score
                        best_prev = prev_idx
                
                pi[j][t_idx] = best_score + emit_prob
                backpointer[j][t_idx] = best_prev
        
        best_final_score = -np.inf
        best_final_tag = 0
        
        for t_idx, tag in enumerate(tags):
            trans_prob = self.get_transition_log_prob(tag, self.STOP_TAG)
            score = pi[n-1][t_idx] + trans_prob
            
            if score > best_final_score:
                best_final_score = score
                best_final_tag = t_idx
        
        best_path = [0] * n
        best_path[n-1] = best_final_tag
        
        for j in range(n-2, -1, -1):
            best_path[j] = backpointer[j+1][best_path[j+1]]
        
        return [tags[idx] for idx in best_path]
    
    def tag_sentence(self, words):
        """Tag a single sentence."""
        tags = self.viterbi_decode(words)
        return list(zip(words, tags))
    
    def tag_sentences(self, sentences):
        tagged = []
        for sentence in sentences:
            words = [word for word, tag in sentence]
            tags = self.viterbi_decode(words)
            tagged.append(list(zip(words, tags)))
        return tagged
    
    def baseline_tag(self, sentences):
        total_tag_counts = Counter()
        for word, tag_counts in self.word_tag_counts.items():
            for tag, count in tag_counts.items():
                total_tag_counts[tag] += count
        
        most_common_tag = total_tag_counts.most_common(1)[0][0] if total_tag_counts else 'NN'
        
        tagged = []
        for sentence in sentences:
            tagged_sent = []
            for word, _ in sentence:
                if word in self.word_tag_counts:
                    best_tag = max(self.word_tag_counts[word].items(), key=lambda x: x[1])[0]
                else:
                    best_tag = most_common_tag
                tagged_sent.append((word, best_tag))
            tagged.append(tagged_sent)
        return tagged
    
    def compute_score(self, words, tags):
        score = 0.0
        prev_tag = self.START_TAG
        
        for word, tag in zip(words, tags):
            score += self.get_transition_log_prob(prev_tag, tag)
            score += self.get_emission_log_prob(tag, word)
            prev_tag = tag
        
        score += self.get_transition_log_prob(prev_tag, self.STOP_TAG)
        return score

def evaluate(test_sentences, tagged_test_sentences):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    print(metrics.classification_report(gold, pred, zero_division=0))
    return gold, pred

def get_accuracy(gold, pred):
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold) * 100


def get_token_tag_tuples(sent):
    return [nltk.tag.str2tuple(t) for t in sent.split()]


def get_tagged_sentences(text):
    sentences = []
    blocks = text.split("======================================")
    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", " ").replace("[", "").replace("]", "").strip()
            if sent:
                sentences.append(sent)
    return sentences


def load_treebank_splits(datadir):
    train = []
    dev = []
    test = []

    for subdir, dirs, files in os.walk(datadir):
        for filename in files:
            if filename.endswith(".pos"):
                filepath = subdir + os.sep + filename
                with open(filepath, "r") as fh:
                    text = fh.read()
                    section = int(subdir.split(os.sep)[-1])
                    
                    if section in range(0, 19):
                        train += get_tagged_sentences(text)
                    elif section in range(19, 22):
                        dev += get_tagged_sentences(text)
                    elif section in range(22, 25):
                        test += get_tagged_sentences(text)

    print(f"Train set size: {len(train)}")
    print(f"Dev set size: {len(dev)}")
    print(f"Test set size: {len(test)}")

    return train, dev, test


def tune_alpha(train_sentences, dev_sentences, alphas=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]):
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    best_alpha = 1.0
    best_accuracy = 0.0
    results = []
    
    for alpha in alphas:
        print(f"\nTesting alpha = {alpha}...")
        tagger = HMMPOSTagger(alpha=alpha)
        tagger.train(train_sentences)
        
        tagged_dev = tagger.tag_sentences(dev_sentences)
        gold = [tag for sent in dev_sentences for _, tag in sent]
        pred = [tag for sent in tagged_dev for _, tag in sent]
        
        accuracy = get_accuracy(gold, pred)
        results.append((alpha, accuracy))
        print(f"  accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} with accuracy: {best_accuracy:.2f}%")
    return best_alpha, results


def compute_confusion_matrix(gold, pred, labels):
    return metrics.confusion_matrix(gold, pred, labels=labels)


def analyze_confusions(cm, labels, top_n=10):
    confusions = []
    for i, true_tag in enumerate(labels):
        for j, pred_tag in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confusions.append((true_tag, pred_tag, cm[i][j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_n]


def main():
    datadir = os.path.join("data", "penn-treeban3-wsj", "wsj")
    
    train_raw, dev_raw, test_raw = load_treebank_splits(datadir)
    
    train_sentences = [get_token_tag_tuples(sent) for sent in train_raw]
    dev_sentences = [get_token_tag_tuples(sent) for sent in dev_raw]
    test_sentences = [get_token_tag_tuples(sent) for sent in test_raw]
    
    print(f"\nProcessed sentences:")
    print(f"  Train: {len(train_sentences)}")
    print(f"  Dev: {len(dev_sentences)}")
    print(f"  Test: {len(test_sentences)}")

    best_alpha, tuning_results = tune_alpha(train_sentences, dev_sentences)
    
    print("\n" + "="*60)
    print(f"TRAINING FINAL MODEL (alpha = {best_alpha})")
    print("="*60)
    
    tagger = HMMPOSTagger(alpha=best_alpha)
    tagger.train(train_sentences)
    
    print("\n" + "="*60)
    print("DEV SET EVALUATION")
    print("="*60)
    
    tagged_dev = tagger.tag_sentences(dev_sentences)
    gold_dev = [tag for sent in dev_sentences for _, tag in sent]
    pred_dev = [tag for sent in tagged_dev for _, tag in sent]
    dev_accuracy = get_accuracy(gold_dev, pred_dev)
    print(f"Dev Accuracy: {dev_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION (HMM Tagger)")
    print("="*60)
    
    tagged_test = tagger.tag_sentences(test_sentences)
    gold_test, pred_test = evaluate(test_sentences, tagged_test)
    test_accuracy = get_accuracy(gold_test, pred_test)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("BASELINE TAGGER EVALUATION")
    print("="*60)
    
    baseline_tagged = tagger.baseline_tag(test_sentences)
    gold_baseline, pred_baseline = evaluate(test_sentences, baseline_tagged)
    baseline_accuracy = get_accuracy(gold_baseline, pred_baseline)
    print(f"\nBaseline Test Accuracy: {baseline_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("COMPARISON: alpha=1 vs best alpha")
    print("="*60)
    
    tagger_alpha1 = HMMPOSTagger(alpha=1.0)
    tagger_alpha1.train(train_sentences)
    tagged_test_alpha1 = tagger_alpha1.tag_sentences(test_sentences)
    pred_alpha1 = [tag for sent in tagged_test_alpha1 for _, tag in sent]
    accuracy_alpha1 = get_accuracy(gold_test, pred_alpha1)
    
    print(f"Test Accuracy with alpha=1.0: {accuracy_alpha1:.2f}%")
    print(f"Test Accuracy with alpha={best_alpha}: {test_accuracy:.2f}%")
    print(f"Improvement: {test_accuracy - accuracy_alpha1:.2f}%")
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    all_tags = sorted(set(gold_test))
    cm = compute_confusion_matrix(gold_test, pred_test, all_tags)
    
    top_confusions = analyze_confusions(cm, all_tags, top_n=15)
    print("-" * 50)
    print(f"{'True Tag':<10} {'Predicted':<10} {'Count':<10}")
    print("-" * 50)
    for true_tag, pred_tag, count in top_confusions:
        print(f"{true_tag:<10} {pred_tag:<10} {count:<10}")
    
    
    single_sent = train_sentences[0]
    debug_tagger = HMMPOSTagger(alpha=0.001)
    debug_tagger.train([single_sent])
    
    words = [w for w, t in single_sent]
    gold_tags = [t for w, t in single_sent]
    pred_tags = debug_tagger.viterbi_decode(words)
    
    print(f"Gold tags: {gold_tags}")
    print(f"Pred tags: {pred_tags}")
    
    gold_score = debug_tagger.compute_score(words, gold_tags)
    pred_score = debug_tagger.compute_score(words, pred_tags)
    print(f"Gold sequence score: {gold_score:.4f}")
    print(f"Pred sequence score: {pred_score:.4f}")
    
    print("\n" + "="*60)
    print(f"\nBaseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"HMM (alpha=1.0) Accuracy: {accuracy_alpha1:.2f}%")
    print(f"HMM (alpha={best_alpha}) Accuracy: {test_accuracy:.2f}%")
    print(f"\nHMM improves over baseline by: {test_accuracy - baseline_accuracy:.2f}%")

if __name__ == "__main__":
    main()