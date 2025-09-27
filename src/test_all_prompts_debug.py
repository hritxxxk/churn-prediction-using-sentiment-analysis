# test_all_prompts_debug.py

import dspy
import re
import concurrent.futures
from prompt_variations import *

# Using local model for now
lm = dspy.LM('ollama/gemma3:270m')
dspy.configure(lm=lm)

def extract_score(response_text):
    # Extract numeric score from response text.
    try:
        print(f"Raw response: {response_text}")
        # If response is a list, get the first element
        if isinstance(response_text, list):
            response_text = response_text[0]
        
        # Try to extract a number
        numbers = re.findall(r'-?\d+\.?\d*', str(response_text))
        if numbers:
            score = float(numbers[0])
            # Clamp to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            print(f"Extracted score: {score}")
            return score
        else:
            print("No numbers found in response")
    except Exception as e:
        print(f"Error extracting score: {e}")
    return 0.0

def test_prompt(prompt_template, review):
    # Test a single prompt with a review.
    try:
        prompt = prompt_template.format(review=review)
        print(f"Prompt: {prompt}")
        response = lm(prompt, max_tokens=30)
        score = extract_score(response)
        return score
    except Exception as e:
        print(f"Error with prompt: {e}")
        return 0.0

def evaluate_prompt(prompt_name, prompt_template, reviews, expected_scores):
    # Evaluate a prompt across all test reviews.
    print(f"\n--- Testing {prompt_name} ---")
    results = []
    errors = 0
    
    for i, (review, expected) in enumerate(zip(reviews, expected_scores)):
        print(f"\nReview {i+1}: {review[:30]}...")
        try:
            predicted = test_prompt(prompt_template, review)
            error = abs(predicted - expected)
            results.append({
                'review_idx': i,
                'review': review[:50] + '...' if len(review) > 50 else review,
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
        except Exception as e:
            errors += 1
            print(f"Error testing {prompt_name} on review {i}: {e}")
    
    # Calculate average error
    avg_error = sum(r['error'] for r in results) / len(results) if results else 1.0
    direction_correct = sum(1 for r in results if (r['predicted'] >= 0) == (r['expected'] >= 0))
    direction_accuracy = direction_correct / len(results) if results else 0.0
    
    return {
        'prompt_name': prompt_name,
        'avg_error': avg_error,
        'direction_accuracy': direction_accuracy,
        'errors': errors,
        'results': results
    }

def main():
    # Test all prompt variations.
    print("Testing all prompt variations with local Gemma model...")
    print("=" * 60)
    
    # Test just one prompt for now to see the output
    result = evaluate_prompt("Basic Direct", PROMPT_BASIC, TEST_REVIEWS[:1], EXPECTED_SCORES[:1])
    print(f"Result: {result}")

if __name__ == "__main__":
    main()