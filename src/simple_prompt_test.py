# simple_prompt_test.py

import dspy
import re

# Using local model for now
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

# Define prompts directly
PROMPT_BASIC = """Analyze the sentiment of the following Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "{review}"

Sentiment score:"""

PROMPT_FEW_SHOT = """Analyze the sentiment of Netflix reviews. Respond ONLY with a decimal number between -1.0 and 1.0 where -1.0 is very negative and 1.0 is very positive.

Examples:
Review: "This movie is absolutely fantastic! I love it so much!"
Sentiment: 0.9

Review: "Terrible film. Complete waste of time and money."
Sentiment: -0.9

Review: "The movie was okay, nothing special."
Sentiment: 0.0

Review: "Outstanding performance by all actors. Highly recommended!"
Sentiment: 1.0

Review: "Not worth watching. Poor plot and bad acting."
Sentiment: -1.0

Now analyze this review:
Review: "{review}"
Sentiment:"""

PROMPT_CONSTRAINED = """You are a sentiment analysis expert. Your task is to analyze Netflix reviews and provide sentiment scores.

Instructions:
1. Read the review carefully
2. Consider both explicit and implicit sentiment
3. Respond ONLY with a single decimal number between -1.0 and 1.0
4. Do not include any other text, explanations, or formatting

Review: "{review}"

Sentiment Score:"""

TEST_REVIEWS = [
    "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging.",
    "Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode.",
    "The movie was okay, nothing special but not bad either. It was entertaining enough to pass the time."
]

EXPECTED_SCORES = [0.9, -0.9, 0.0]

def extract_score(response_text):
    # Extract numeric score from response text.
    try:
        # If response is a list, get the first element
        if isinstance(response_text, list):
            response_text = response_text[0]
        
        # Try to extract a number
        numbers = re.findall(r'-?\d+\.?\d*', str(response_text))
        if numbers:
            score = float(numbers[0])
            # Clamp to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            return score
    except Exception as e:
        print(f"Error extracting score: {e}")
    return 0.0

def test_prompt(prompt_template, review):
    # Test a single prompt with a review.
    try:
        prompt = prompt_template.format(review=review)
        response = lm(prompt, max_tokens=30)
        score = extract_score(response)
        return score
    except Exception as e:
        print(f"Error with prompt: {e}")
        return 0.0

def evaluate_prompt(prompt_name, prompt_template):
    # Evaluate a prompt across test reviews.
    print(f"Testing {prompt_name}...")
    results = []
    
    for i, (review, expected) in enumerate(zip(TEST_REVIEWS, EXPECTED_SCORES)):
        predicted = test_prompt(prompt_template, review)
        error = abs(predicted - expected)
        results.append({
            'review_idx': i,
            'expected': expected,
            'predicted': predicted,
            'error': error
        })
        print(f"  Review {i+1}: Expected={expected:.1f}, Predicted={predicted:.1f}")
    
    # Calculate average error
    avg_error = sum(r['error'] for r in results) / len(results)
    direction_correct = sum(1 for r in results if (r['predicted'] >= 0) == (r['expected'] >= 0))
    direction_accuracy = direction_correct / len(results)
    
    return {
        'prompt_name': prompt_name,
        'avg_error': avg_error,
        'direction_accuracy': direction_accuracy,
        'results': results
    }

def main():
    # Test prompt variations.
    print("Testing prompt variations with local Gemma model...")
    print("=" * 60)
    
    # Define prompts to test
    prompts = [
        ("Basic Direct", PROMPT_BASIC),
        ("Few-Shot", PROMPT_FEW_SHOT),
        ("Constrained", PROMPT_CONSTRAINED)
    ]
    
    evaluations = []
    for name, template in prompts:
        result = evaluate_prompt(name, template)
        evaluations.append(result)
    
    # Sort by average error (lower is better)
    evaluations.sort(key=lambda x: x['avg_error'])
    
    # Print results
    print("\n" + "=" * 60)
    print("PROMPT EVALUATION RESULTS (sorted by average error)")
    print("=" * 60)
    
    for i, eval_result in enumerate(evaluations):
        print(f"\n{i+1}. {eval_result['prompt_name']}")
        print(f"   Average Error: {eval_result['avg_error']:.3f}")
        print(f"   Direction Accuracy: {eval_result['direction_accuracy']:.1%}")
        
        # Show predictions
        print("   Predictions:")
        for res in eval_result['results']:
            print(f"     Review {res['review_idx']+1}: Expected={res['expected']:.1f}, Predicted={res['predicted']:.1f}")
    
    # Identify best prompt
    best_prompt = evaluations[0]
    print(f"\n" + "=" * 60)
    print(f"BEST PERFORMING PROMPT: {best_prompt['prompt_name']}")
    print(f"Average Error: {best_prompt['avg_error']:.3f}")
    print(f"Direction Accuracy: {best_prompt['direction_accuracy']:.1%}")
    print("=" * 60)

if __name__ == "__main__":
    main()