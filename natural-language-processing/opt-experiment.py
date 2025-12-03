import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


class OPTExperiment:
    def __init__(self, model_name="facebook/opt-1.3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, num_return_sequences=1):
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
            generated_text = text[len(prompt):].strip()
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def extract_answer(self, generated_text):
        lines = generated_text.split('\n')
        if lines:
            answer = lines[0].strip()
            answer = answer.split('.')[0].strip()
            answer = answer.split('Question:')[0].strip()
            return answer
        return generated_text


def experiment_1_zero_shot_qa(model):
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

        generated = model.generate_text(prompt, max_new_tokens=30, temperature=0.3)
        answer = model.extract_answer(generated[0])
        
        print(f"Generated Answer: {answer}")
        
        results.append({
            'question': question,
            'answer': answer,
            'full_output': generated[0]
        })
    expected_answers = [
        "Samuel Webber",
        "1803",
        "Henry Ware",
        "1804"
    ]
    
    print(f"{'='*60}")
    print("Zero Shot")
    for i, (result, expected) in enumerate(zip(results, expected_answers), 1):
        print(f"\n{i}. {result['question']}")
        print(f"   Model Answer: {result['answer']}")
        print(f"   Expected: {expected}")
    return results


def experiment_2_few_shot_qa(model):
    print("=" * 80)
    print("Few Shot")
    
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
    
    
    # Generate answer
    generated = model.generate_text(prompt, max_new_tokens=30, temperature=0.3)
    answer = model.extract_answer(generated[0])
    
    print(f"Generated Answer: {answer}")
    print(f"Expected Answer: Samuel Webber")
    
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
    print("Sentiment Analysis")
    print("=" * 80)
    
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



def main():
    model = OPTExperiment("facebook/opt-1.3b")

    zero_shot_results = experiment_1_zero_shot_qa(model)

    few_shot_results = experiment_2_few_shot_qa(model)
    
    sentiment_zero, sentiment_few = experiment_3_sentiment_analysis(model)
    
if __name__ == "__main__":
    main()