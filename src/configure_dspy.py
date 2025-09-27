# modular_version/configure_dspy.py

import argparse
import os
import dspy

def configure_openai():
    """Configure DSPy with OpenAI models."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    try:
        lm = dspy.LM('openai/gpt-4o-mini')
        dspy.configure(lm=lm)
        print("Successfully configured DSPy with OpenAI GPT-4o-mini")
        return True
    except Exception as e:
        print(f"Error configuring OpenAI: {e}")
        return False

def configure_anthropic():
    """Configure DSPy with Anthropic models."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY=your_api_key_here")
        return False
    
    try:
        lm = dspy.LM('anthropic/claude-3-haiku')
        dspy.configure(lm=lm)
        print("Successfully configured DSPy with Anthropic Claude-3-Haiku")
        return True
    except Exception as e:
        print(f"Error configuring Anthropic: {e}")
        return False

def configure_google():
    """Configure DSPy with Google Gemini models."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set it with: export GOOGLE_API_KEY=your_api_key_here")
        print("Get your API key from: https://ai.google.dev/")
        return False
    
    try:
        lm = dspy.LM('google/gemini-pro')
        dspy.configure(lm=lm)
        print("Successfully configured DSPy with Google Gemini Pro")
        return True
    except Exception as e:
        print(f"Error configuring Google Gemini: {e}")
        return False

def configure_local():
    """Configure DSPy with a local model via Ollama."""
    try:
        # Using the smaller gemma3:270m model
        lm = dspy.LM('ollama/gemma3:270m')
        dspy.configure(lm=lm)
        print("Successfully configured DSPy with local Gemma3 (270M) model")
        return True
    except Exception as e:
        print(f"Error configuring local model: {e}")
        print("Make sure Ollama is installed and running, and the model is downloaded:")
        print("  Install Ollama: https://ollama.com/")
        print("  Download model: ollama pull gemma3:270m")
        return False

def main():
    """Main configuration function."""
    parser = argparse.ArgumentParser(description="Configure DSPy for Netflix Churn Prediction")
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'google', 'local'], 
                        default='openai', help='Language model provider')
    parser.add_argument('--model', type=str, help='Specific model to use')
    
    args = parser.parse_args()
    
    print("Configuring DSPy for Netflix Churn Prediction System")
    print("=" * 55)
    
    success = False
    if args.provider == 'openai':
        success = configure_openai()
    elif args.provider == 'anthropic':
        success = configure_anthropic()
    elif args.provider == 'google':
        success = configure_google()
    elif args.provider == 'local':
        success = configure_local()
    
    if success:
        print("\nDSPy is now configured and ready to use!")
        print("You can now run the main analysis with DSPy integration.")
    else:
        print("\nFailed to configure DSPy. Using fallback methods.")

if __name__ == "__main__":
    main()