# prompt_variations.py

# Different prompt variations for sentiment analysis

# 1. Basic direct prompt
PROMPT_BASIC = """Analyze the sentiment of the following Netflix review and respond ONLY with a number between -1.0 and 1.0:
-1.0 = very negative sentiment
0.0 = neutral sentiment  
1.0 = very positive sentiment

Review: "{review}"

Sentiment score:"""

# 2. Few-shot prompt with examples
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

# 3. Chain-of-thought prompt
PROMPT_COT = """Analyze the sentiment of the following Netflix review step by step, then provide a final sentiment score.

Review: "{review}"

Step 1: Identify positive words or phrases
Step 2: Identify negative words or phrases  
Step 3: Consider the overall tone
Step 4: Assign a sentiment score between -1.0 (very negative) and 1.0 (very positive)

Final Sentiment Score:"""

# 4. Instructional prompt with constraints
PROMPT_CONSTRAINED = """You are a sentiment analysis expert. Your task is to analyze Netflix reviews and provide sentiment scores.

Instructions:
1. Read the review carefully
2. Consider both explicit and implicit sentiment
3. Respond ONLY with a single decimal number between -1.0 and 1.0
4. Do not include any other text, explanations, or formatting

Review: "{review}"

Sentiment Score:"""

# 5. Role-based prompt
PROMPT_ROLE = """You are a professional movie critic with expertise in sentiment analysis. Analyze the emotional tone of this Netflix review and provide a quantitative assessment.

Scale: -1.0 (extremely negative) to 1.0 (extremely positive)

Review: "{review}"

Your quantitative assessment:"""

# 6. Structured format prompt
PROMPT_STRUCTURED = """SENTIMENT ANALYSIS TASK
=======================

INPUT: Netflix review text
OUTPUT: Single decimal number between -1.0 and 1.0

Review: "{review}"

Analysis:
1. Positive indicators: [list key positive elements]
2. Negative indicators: [list key negative elements]
3. Overall assessment: [brief summary]

Numerical sentiment score:"""

# Test reviews
TEST_REVIEWS = [
    "This show is absolutely amazing! I've been binge-watching it every day. The characters are well-developed and the plot is engaging.",
    "Terrible show. Waste of time. Poor acting and boring plot. I stopped watching after the first episode.",
    "The movie was okay, nothing special but not bad either. It was entertaining enough to pass the time.",
    "Outstanding performance by all actors. The storyline kept me hooked from beginning to end. A masterpiece!",
    "Not worth watching. The story doesn't make sense and the characters are poorly written. Save your time and money."
]

# Expected scores (approximate)
EXPECTED_SCORES = [0.9, -0.9, 0.0, 0.95, -0.85]