"""
Large Language Model Experiments with OPT-1.3b
Assignment 2 - Part 3: Experiments with Language Models
NLP 201: Natural Language Processing I

This module implements experiments with the OPT-1.3b model from Meta/Facebook
for zero-shot and few-shot learning on question answering and other NLP tasks.

Alternative to GPT-3 as the original API access has changed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


class OPTExperiment:
    """
    Class for running experiments with OPT-1.3b model.
    """
    
    def __init__(self, model_name="facebook/opt-1.3b"):
        """
        Initialize the OPT model and tokenizer.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print("=" * 80)
        print(f"Loading {model_name} model...")
        print("This may take a few minutes on first run (downloads ~2.5GB)")
        print("=" * 80)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✓ Using CPU (slower but works)")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded successfully!")
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, num_return_sequences=1):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of completions to generate
            
        Returns:
            List of generated texts
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the output
            generated_text = text[len(prompt):].strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def extract_answer(self, generated_text):
        """
        Extract the answer from generated text.
        Often the model generates extra text after the answer.
        """
        # Try to extract just the answer (first sentence or line)
        lines = generated_text.split('\n')
        if lines:
            answer = lines[0].strip()
            # Remove common artifacts
            answer = answer.split('.')[0].strip()
            answer = answer.split('Question:')[0].strip()
            return answer
        return generated_text


def experiment_1_zero_shot_qa(model):
    """
    Deliverable 2: Zero-shot question answering on Harvard passage.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ZERO-SHOT QUESTION ANSWERING")
    print("=" * 80)
    
    # The Harvard passage from the assignment
    passage = """When the Hollis Professor of Divinity David Tappan died in 1803 and the president of Harvard Joseph Willard died a year later, in 1804, a struggle broke out over their replacements. Henry Ware was elected to the chair in 1805, and the liberal Samuel Webber was appointed to the presidency of Harvard two years later."""
    
    questions = [
        "Who succeeded Joseph Willard as president?",
        "When did David Tappan die?",
        "Who was elected to the chair in 1805?",
        "What year did Joseph Willard die?",
    ]
    
    results = []
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'-'*60}")
        
        prompt = f"""Read the passage and answer the question.

{passage}

Question: {question}
Answer:"""
        
        # Generate answer
        generated = model.generate_text(prompt, max_new_tokens=30, temperature=0.3)
        answer = model.extract_answer(generated[0])
        
        print(f"Generated Answer: {answer}")
        
        results.append({
            'question': question,
            'answer': answer,
            'full_output': generated[0]
        })
    
    # Expected answers for comparison
    expected_answers = [
        "Samuel Webber",
        "1803",
        "Henry Ware",
        "1804"
    ]
    
    print(f"\n{'='*60}")
    print("SUMMARY - Zero-Shot Results:")
    print(f"{'='*60}")
    for i, (result, expected) in enumerate(zip(results, expected_answers), 1):
        print(f"\n{i}. {result['question']}")
        print(f"   Model Answer: {result['answer']}")
        print(f"   Expected: {expected}")
        # Simple check if expected answer is in the response
        correct = expected.lower() in result['answer'].lower()
        print(f"   Status: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
    
    return results


def experiment_2_few_shot_qa(model):
    """
    Deliverable 3: Few-shot question answering with examples.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: FEW-SHOT QUESTION ANSWERING")
    print("=" * 80)
    
    # Few-shot prompt with examples
    prompt = """Read the passage and answer the following questions.

In the late 17th century, Robert Boyle proved that air is necessary for combustion. English chemist John Mayow (1641-1679) refined this work by showing that fire requires only a part of air that he called spiritus nitroaereus or just nitroaereus.

Question: Who proved that air is necessary for combustion?
Answer: Robert Boyle

Question: John Mayow died in what year?
Answer: 1679

Read the passage and answer the following questions.

When the Hollis Professor of Divinity David Tappan died in 1803 and the president of Harvard Joseph Willard died a year later, in 1804, a struggle broke out over their replacements. Henry Ware was elected to the chair in 1805, and the liberal Samuel Webber was appointed to the presidency of Harvard two years later.

Question: Who succeeded Joseph Willard as president?
Answer:"""
    
    print("Using few-shot prompt with examples...")
    print(f"\n{'-'*60}")
    
    # Generate answer
    generated = model.generate_text(prompt, max_new_tokens=30, temperature=0.3)
    answer = model.extract_answer(generated[0])
    
    print(f"Generated Answer: {answer}")
    print(f"Expected Answer: Samuel Webber")
    
    correct = "samuel webber" in answer.lower()
    print(f"Status: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
    
    # Try more questions with few-shot
    print(f"\n{'='*60}")
    print("Testing additional questions with few-shot learning:")
    print(f"{'='*60}")
    
    additional_questions = [
        "When did David Tappan die?",
        "Who was elected to the chair in 1805?",
    ]
    
    results = [{'question': "Who succeeded Joseph Willard as president?", 'answer': answer}]
    
    for question in additional_questions:
        print(f"\n{'-'*60}")
        print(f"Question: {question}")
        print(f"{'-'*60}")
        
        prompt_additional = """Read the passage and answer the following questions.

In the late 17th century, Robert Boyle proved that air is necessary for combustion. English chemist John Mayow (1641-1679) refined this work by showing that fire requires only a part of air that he called spiritus nitroaereus or just nitroaereus.

Question: Who proved that air is necessary for combustion?
Answer: Robert Boyle

Question: John Mayow died in what year?
Answer: 1679

Read the passage and answer the following questions.

When the Hollis Professor of Divinity David Tappan died in 1803 and the president of Harvard Joseph Willard died a year later, in 1804, a struggle broke out over their replacements. Henry Ware was elected to the chair in 1805, and the liberal Samuel Webber was appointed to the presidency of Harvard two years later.

Question: """ + question + """
Answer:"""
        
        generated = model.generate_text(prompt_additional, max_new_tokens=30, temperature=0.3)
        answer = model.extract_answer(generated[0])
        
        print(f"Generated Answer: {answer}")
        results.append({'question': question, 'answer': answer})
    
    return results


def experiment_3_sentiment_analysis(model):
    """
    Deliverable 5: Try another task - Sentiment Analysis
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: SENTIMENT ANALYSIS TASK")
    print("=" * 80)
    
    print("\n--- Zero-Shot Sentiment Analysis ---")
    
    test_sentences = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The food was terrible and the service was even worse.",
        "It was okay, nothing special but not bad either.",
        "I'm so happy with my new phone, it works perfectly!",
        "What a waste of money, I'm very disappointed.",
    ]
    
    zero_shot_results = []
    
    for sentence in test_sentences:
        print(f"\n{'-'*60}")
        print(f"Sentence: {sentence}")
        
        prompt = f"""Analyze the sentiment of the following sentence as Positive, Negative, or Neutral.

Sentence: {sentence}
Sentiment:"""
        
        generated = model.generate_text(prompt, max_new_tokens=10, temperature=0.3)
        sentiment = model.extract_answer(generated[0])
        
        print(f"Predicted Sentiment: {sentiment}")
        zero_shot_results.append({'sentence': sentence, 'sentiment': sentiment})
    
    print("\n" + "=" * 60)
    print("--- Few-Shot Sentiment Analysis ---")
    
    few_shot_results = []
    
    for sentence in test_sentences:
        print(f"\n{'-'*60}")
        print(f"Sentence: {sentence}")
        
        prompt = f"""Analyze the sentiment of the following sentences as Positive, Negative, or Neutral.

Sentence: The product exceeded my expectations.
Sentiment: Positive

Sentence: This was a complete disaster.
Sentiment: Negative

Sentence: It's acceptable, neither good nor bad.
Sentiment: Neutral

Sentence: {sentence}
Sentiment:"""
        
        generated = model.generate_text(prompt, max_new_tokens=10, temperature=0.3)
        sentiment = model.extract_answer(generated[0])
        
        print(f"Predicted Sentiment: {sentiment}")
        few_shot_results.append({'sentence': sentence, 'sentiment': sentiment})
    
    return zero_shot_results, few_shot_results


def experiment_4_summarization(model):
    """
    Deliverable 5: Try another task - Summarization
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SUMMARIZATION TASK")
    print("=" * 80)
    
    text = """Climate change is one of the most pressing issues facing humanity today. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause of recent climate change. The effects are already being felt around the world, from more frequent hurricanes to prolonged droughts. Many countries are now taking action to reduce carbon emissions and transition to renewable energy sources."""
    
    print("\n--- Zero-Shot Summarization ---")
    print(f"\nOriginal Text:\n{text}")
    
    prompt = f"""Summarize the following text in one sentence.

Text: {text}

Summary:"""
    
    generated = model.generate_text(prompt, max_new_tokens=50, temperature=0.5)
    summary = model.extract_answer(generated[0])
    
    print(f"\nGenerated Summary:\n{summary}")
    
    print("\n" + "=" * 60)
    print("--- Few-Shot Summarization ---")
    
    prompt_few_shot = """Summarize the following texts in one sentence.

Text: The Amazon rainforest is home to millions of species. Deforestation threatens this biodiversity. Conservation efforts are crucial for protecting these ecosystems.
Summary: The Amazon rainforest's biodiversity is threatened by deforestation, requiring conservation efforts.

Text: Artificial intelligence is transforming industries worldwide. Machine learning enables computers to learn from data. Applications range from healthcare to autonomous vehicles.
Summary: Artificial intelligence and machine learning are revolutionizing various industries from healthcare to transportation.

Text: Climate change is one of the most pressing issues facing humanity today. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause of recent climate change. The effects are already being felt around the world, from more frequent hurricanes to prolonged droughts. Many countries are now taking action to reduce carbon emissions and transition to renewable energy sources.
Summary:"""
    
    generated = model.generate_text(prompt_few_shot, max_new_tokens=50, temperature=0.5)
    summary_few_shot = model.extract_answer(generated[0])
    
    print(f"\nGenerated Summary (Few-Shot):\n{summary_few_shot}")
    
    return summary, summary_few_shot


def experiment_5_comparison(model):
    """
    Compare zero-shot vs few-shot performance.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: COMPARING ZERO-SHOT VS FEW-SHOT")
    print("=" * 80)
    
    print("""
Analysis of Results:

1. Question Answering:
   - Zero-shot: The model attempts to answer but may lack the specific format
   - Few-shot: Providing examples helps the model understand the expected format
   - OPT-1.3b may struggle with complex reasoning compared to larger models

2. Sentiment Analysis:
   - Zero-shot: Can classify basic sentiments but may be inconsistent
   - Few-shot: Examples help standardize the output format (Positive/Negative/Neutral)
   - Performance improves with examples showing the exact format

3. Summarization:
   - Zero-shot: Generates summaries but may be verbose or off-target
   - Few-shot: Examples help constrain length and focus
   - Few-shot typically produces more concise, better-structured summaries

4. General Observations:
   - OPT-1.3b (1.3 billion parameters) is smaller than GPT-3 (175 billion)
   - Performance is reasonable but not as strong as larger models
   - Few-shot learning consistently improves results
   - Temperature affects creativity vs consistency
   - The model sometimes generates extra text beyond the answer

5. Comparison to GPT-3:
   - OPT-1.3b: Open-source, smaller, more accessible
   - GPT-3: Larger, more capable, but requires API access
   - OPT-1.3b demonstrates similar patterns but with lower accuracy
   - Both benefit significantly from few-shot examples
""")


def main():
    """
    Main function to run all experiments for Part 3.
    """
    print("=" * 80)
    print("PART 3: LANGUAGE MODEL EXPERIMENTS WITH OPT-1.3B")
    print("Alternative Implementation (Using OPT-1.3b instead of GPT-3)")
    print("=" * 80)
    
    # Initialize model
    model = OPTExperiment("facebook/opt-1.3b")
    
    # Run experiments
    
    # Deliverable 2: Zero-shot question answering
    zero_shot_results = experiment_1_zero_shot_qa(model)
    
    # Deliverable 3: Few-shot question answering
    few_shot_results = experiment_2_few_shot_qa(model)
    
    # Deliverable 5: Other tasks
    sentiment_zero, sentiment_few = experiment_3_sentiment_analysis(model)
    summary_zero, summary_few = experiment_4_summarization(model)
    
    # Analysis and comparison
    experiment_5_comparison(model)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY FOR REPORT")
    print("=" * 80)
    
    print("""
For your report, include:

1. Model Information:
   - Model: facebook/opt-1.3b (Meta's Open Pretrained Transformer)
   - Size: 1.3 billion parameters
   - Alternative to GPT-3 for educational purposes
   - Open-source and freely available

2. Zero-Shot Question Answering Results:
   - Include the table of questions, model answers, and expected answers
   - Discuss accuracy and failure modes

3. Few-Shot Question Answering Results:
   - Show how examples improve performance
   - Compare to zero-shot results

4. Additional Task (Sentiment Analysis):
   - Show zero-shot and few-shot results
   - Discuss which performs better and why

5. Additional Task (Summarization):
   - Demonstrate the model's ability to condense information
   - Compare zero-shot vs few-shot quality

6. Analysis:
   - Discuss the benefits of few-shot learning
   - Compare OPT-1.3b capabilities to what you'd expect from GPT-3
   - Explain how model size affects performance
   - Discuss practical implications of using open-source vs proprietary models

7. Limitations:
   - OPT-1.3b is smaller than GPT-3 (1.3B vs 175B parameters)
   - May have lower accuracy on complex reasoning tasks
   - Sometimes generates extra text or goes off-topic
   - Benefits significantly from prompt engineering
""")
    
    print("\n" + "=" * 80)
    print("Experiments completed! Use the output above for your report.")
    print("=" * 80)


if __name__ == "__main__":
    main()