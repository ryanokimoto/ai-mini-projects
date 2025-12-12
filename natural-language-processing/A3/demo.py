#!/usr/bin/env python
"""
Demo script for HMM POS Tagger
Demonstrates functionality with small examples before running on full dataset
"""

from hmm_pos_tagger import HMMPOSTagger, BaselineTagger
import numpy as np

def demo_small_example():
    """Demo with a tiny training set for debugging"""
    print("="*80)
    print("DEMO: Small Example")
    print("="*80)
    
    # Minimal training data
    mini_train = [
        "The/DT cat/NN sat/VBD",
        "The/DT dog/NN ran/VBD",
        "A/DT cat/NN sleeps/VBZ",
        "The/DT mouse/NN ran/VBD",
    ]
    
    # Test sentence
    test_sent = [("The", "DT"), ("cat", "NN"), ("ran", "VBD")]
    
    print(f"\nTraining on {len(mini_train)} sentences...")
    print("Training sentences:")
    for sent in mini_train:
        print(f"  {sent}")
    
    # Train HMM
    tagger = HMMPOSTagger(alpha=1.0)
    tagger.train(mini_train)
    
    print(f"\nVocabulary size: {len(tagger.vocab)}")
    print(f"Tagset: {sorted(tagger.tagset - {tagger.STOP})}")
    
    # Show some probabilities
    print("\nSample Transition Probabilities (log):")
    print(f"  P(NN | DT) = {np.exp(tagger._get_transition_prob('DT', 'NN')):.4f}")
    print(f"  P(VBD | NN) = {np.exp(tagger._get_transition_prob('NN', 'VBD')):.4f}")
    
    print("\nSample Emission Probabilities (log):")
    print(f"  P('cat' | NN) = {np.exp(tagger._get_emission_prob('NN', 'cat')):.4f}")
    print(f"  P('ran' | VBD) = {np.exp(tagger._get_emission_prob('VBD', 'ran')):.4f}")
    
    # Test
    print(f"\nTest sentence: {' '.join([w for w, t in test_sent])}")
    print(f"Gold tags: {' '.join([t for w, t in test_sent])}")
    
    tagged = tagger.viterbi_decode(test_sent)
    pred_tags = [t for w, t in tagged]
    
    print(f"Predicted tags: {' '.join(pred_tags)}")
    
    correct = sum(1 for i in range(len(test_sent)) if test_sent[i][1] == pred_tags[i])
    accuracy = correct / len(test_sent)
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(test_sent)})")


def demo_viterbi_steps():
    """Show Viterbi algorithm step-by-step"""
    print("\n" + "="*80)
    print("DEMO: Viterbi Algorithm Steps")
    print("="*80)
    
    # Very simple example
    mini_train = [
        "DT/DT NN/NN",
        "DT/DT VB/VB",
        "NN/NN VB/VB",
    ]
    
    tagger = HMMPOSTagger(alpha=0.5)
    tagger.train(mini_train)
    
    # Test
    words = ["DT", "NN"]
    print(f"\nInput: {' '.join(words)}")
    print("\nViterbi forward pass:")
    
    # Manual Viterbi to show steps
    n = len(words)
    viterbi = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]
    
    # Step 0
    print(f"\nStep 0 (word='{words[0]}'):")
    for tag in ['DT', 'NN', 'VB']:
        emission = tagger._get_emission_prob(tag, words[0])
        transition = tagger._get_transition_prob(tagger.START, tag)
        viterbi[0][tag] = transition + emission
        backpointer[0][tag] = tagger.START
        print(f"  π₀({tag}) = {viterbi[0][tag]:.4f} (from {tagger.START})")
    
    # Step 1
    print(f"\nStep 1 (word='{words[1]}'):")
    for tag in ['DT', 'NN', 'VB']:
        max_prob = float('-inf')
        best_prev = None
        emission = tagger._get_emission_prob(tag, words[1])
        
        for prev_tag in viterbi[0]:
            transition = tagger._get_transition_prob(prev_tag, tag)
            prob = viterbi[0][prev_tag] + transition + emission
            if prob > max_prob:
                max_prob = prob
                best_prev = prev_tag
        
        viterbi[1][tag] = max_prob
        backpointer[1][tag] = best_prev
        print(f"  π₁({tag}) = {viterbi[1][tag]:.4f} (from {best_prev})")
    
    # Final
    print(f"\nFinal step (transition to STOP):")
    max_prob = float('-inf')
    best_final = None
    for tag in viterbi[n-1]:
        transition = tagger._get_transition_prob(tag, tagger.STOP)
        prob = viterbi[n-1][tag] + transition
        print(f"  Final score via {tag}: {prob:.4f}")
        if prob > max_prob:
            max_prob = prob
            best_final = tag
    
    # Backtrack
    path = [best_final]
    for i in range(n-1, 0, -1):
        path.insert(0, backpointer[i][path[0]])
    
    print(f"\nBest path: {' → '.join([tagger.START] + path + [tagger.STOP])}")
    print(f"Best score: {max_prob:.4f}")
    print(f"\nTagged output: {' '.join([f'{words[i]}/{path[i]}' for i in range(n)])}")


def demo_unknown_words():
    """Show handling of unknown words"""
    print("\n" + "="*80)
    print("DEMO: Unknown Word Handling")
    print("="*80)
    
    mini_train = [
        "The/DT cat/NN sleeps/VBZ",
        "A/DT dog/NN runs/VBZ",
        "The/DT bird/NN flies/VBZ",
    ]
    
    tagger = HMMPOSTagger(alpha=1.0)
    tagger.train(mini_train)
    
    print(f"\nKnown words: {sorted(tagger.vocab)}")
    
    # Test with unknown word
    test_words = ["The", "zebra", "runs"]  # "zebra" is unknown
    print(f"\nTest: {' '.join(test_words)}")
    print(f"  'zebra' is UNKNOWN (not in training)")
    
    tagged = tagger.viterbi_decode(test_words)
    print(f"\nTagged: {' '.join([f'{w}/{t}' for w, t in tagged])}")
    print("\nNote: Unknown word 'zebra' gets tagged using:")
    print("  1. <UNK> emission probabilities from each tag")
    print("  2. Transition probabilities from previous tag")
    print("  3. Viterbi algorithm to find best overall sequence")


def demo_baseline_comparison():
    """Compare HMM with baseline"""
    print("\n" + "="*80)
    print("DEMO: HMM vs Baseline")
    print("="*80)
    
    train = [
        "The/DT cat/NN sleeps/VBZ ./.",
        "A/DT dog/NN runs/VBZ ./.",
        "The/DT bird/NN flies/VBZ ./.",
        "The/DT mouse/NN squeaks/VBZ ./.",
        "I/PRP can/MD run/VB ./.",
    ]
    
    test = [
        [("The", "DT"), ("cat", "NN"), ("runs", "VBZ"), (".", ".")],
        [("I", "PRP"), ("can", "MD"), ("run", "VB"), (".", ".")],
    ]
    
    print(f"\nTraining on {len(train)} sentences")
    
    # Baseline
    baseline = BaselineTagger()
    baseline.train(train)
    baseline_tagged = baseline.tag_sentences(test)
    
    # HMM
    hmm = HMMPOSTagger(alpha=1.0)
    hmm.train(train)
    hmm_tagged = hmm.tag_sentences(test)
    
    print("\nResults:")
    for i, test_sent in enumerate(test):
        gold = [t for w, t in test_sent]
        baseline_pred = [t for w, t in baseline_tagged[i]]
        hmm_pred = [t for w, t in hmm_tagged[i]]
        
        print(f"\nSentence {i+1}: {' '.join([w for w, t in test_sent])}")
        print(f"  Gold:     {' '.join(gold)}")
        print(f"  Baseline: {' '.join(baseline_pred)}")
        print(f"  HMM:      {' '.join(hmm_pred)}")
        
        baseline_correct = sum(1 for j in range(len(gold)) if gold[j] == baseline_pred[j])
        hmm_correct = sum(1 for j in range(len(gold)) if gold[j] == hmm_pred[j])
        
        print(f"  Baseline accuracy: {baseline_correct}/{len(gold)}")
        print(f"  HMM accuracy: {hmm_correct}/{len(gold)}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HMM POS TAGGER - DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows the tagger working on small examples")
    print("For full Penn Treebank results, run: python hmm_pos_tagger.py")
    print("="*80)
    
    demo_small_example()
    demo_viterbi_steps()
    demo_unknown_words()
    demo_baseline_comparison()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo run on full dataset:")
    print("  python hmm_pos_tagger.py")
    print("\nMake sure your data is in: data/penn-treebank3-wsj/wsj/")
    print("="*80 + "\n")
