# Usage Guide

## Getting Started

### Quick Start
To run a basic analysis with sample data:

```bash
python -m src.main --data data/small_sample.csv
```

### Basic Python Usage
```python
from src.analysis.analyzer import NetflixAnalyzer

# Initialize analyzer
analyzer = NetflixAnalyzer("data/small_sample.csv")

# Run analysis
results = analyzer.run_analysis()

# Save results
analyzer.save_preprocessed_data("output/results.csv")
```

## Command Line Interface

### Main Script Options
```bash
python -m src.main --help
```

Options:
- `--data PATH`: Path to input data file (CSV format)
- `--output PATH`: Path to output results file
- `--advanced-nlp`: Use advanced NLP techniques
- `--advanced-sentiment`: Use transformer-based sentiment analysis
- `--dspy-sentiment`: Use DSPy-based sentiment analysis

### Examples

#### Basic Analysis
```bash
python -m src.main --data data/my_reviews.csv --output output/my_results.csv
```

#### Advanced Analysis
```bash
python -m src.main --data data/my_reviews.csv --advanced-nlp --advanced-sentiment
```

#### With Foundation Models
```bash
python -m src.main --data data/my_reviews.csv --dspy-sentiment
```

## Python API Usage

### 1. Basic Analysis
```python
from src.analysis.analyzer import NetflixAnalyzer

# Initialize analyzer with data file
analyzer = NetflixAnalyzer("data/netflix_reviews.csv")

# Run basic analysis
results = analyzer.run_analysis(
    use_advanced_nlp=False,
    use_advanced_sentiment=False,
    use_dspy_sentiment=False
)

# Access results
print(f"Model accuracy: {results['model_metrics']['accuracy']:.4f}")
```

### 2. Advanced Analysis
```python
# Run analysis with advanced NLP
results = analyzer.run_analysis(
    use_advanced_nlp=True,      # Use lemmatization and advanced text processing
    use_advanced_sentiment=True, # Use transformer-based sentiment analysis
    use_dspy_sentiment=False    # Don't use DSPy foundation models
)
```

### 3. Foundation Model Analysis
```python
# Run analysis with DSPy foundation models
results = analyzer.run_analysis(
    use_advanced_nlp=False,
    use_advanced_sentiment=False,
    use_dspy_sentiment=True     # Use DSPy-based sentiment analysis
)
```

## Analysis Pipeline Components

### 1. Data Processing
```python
from src.data_processing.data_processor import DataProcessor

processor = DataProcessor()
df = processor.load_data("data/reviews.csv")
df_cleaned = processor.clean_data(df)
df_features = processor.create_features(df_cleaned)
```

### 2. Model Training
```python
from src.model_training.model_trainer import ModelTrainer

trainer = ModelTrainer()
model, X_train, X_test, y_train, y_test = trainer.train_model(df_features)
```

### 3. Analysis and Visualization
```python
from src.visualization.explainer import Explainer

explainer = Explainer()
explainer_obj, shap_values = explainer.explain_with_shap(model, X_train, X_test)
lime_explanation = explainer.explain_with_lime(model, X_train, X_test)
```

## Configuration Options

### Environment Variables
Set these in your `.env` file:

```bash
# Google API Key for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Runtime Configuration
```python
import os
from src.config.settings import GOOGLE_API_KEY, RANDOM_STATE

# Override settings
os.environ['GOOGLE_API_KEY'] = 'your_temporary_key'

# Use in analysis
analyzer = NetflixAnalyzer("data/reviews.csv", random_state=42)
```

## Output Files

### Generated Files
The system creates several output files:

1. **Processed Data**: `output/results.csv`
   - Cleaned and processed review data
   - Engineered features
   - Churn risk predictions

2. **Model Reports**: `output/model_report.txt`
   - Performance metrics
   - Feature importance
   - Cross-validation results

3. **Visualizations**: `output/plots/`
   - Clustering visualizations
   - Feature distributions
   - Model performance charts

4. **Explanation Files**: `output/explanations/`
   - SHAP explanations
   - LIME local explanations
   - Feature importance plots

## Customization Options

### 1. Feature Engineering Parameters
```python
# Modify feature engineering thresholds
analyzer = NetflixAnalyzer(
    "data/reviews.csv",
    satisfaction_quantile=0.3,  # Adjust satisfaction threshold
    sentiment_threshold=-0.3,   # Adjust sentiment threshold
    score_threshold=2          # Adjust score threshold
)
```

### 2. Model Parameters
```python
# Customize model training
from src.model_training.model_trainer import ModelTrainer

trainer = ModelTrainer(
    n_estimators=100,    # Number of trees
    max_depth=15,        # Maximum tree depth
    min_samples_split=10 # Minimum samples to split
)
```

### 3. Clustering Parameters
```python
# Customize clustering
trainer = ModelTrainer(
    n_clusters=5,                # Number of clusters
    pca_variance_threshold=0.95  # PCA variance threshold
)
```

## Analysis Results Interpretation

### 1. Model Performance Metrics
```python
results = analyzer.run_analysis()

# Access performance metrics
accuracy = results['model_metrics']['accuracy']
precision = results['model_metrics']['precision']
recall = results['model_metrics']['recall']
f1_score = results['model_metrics']['f1_score']
```

### 2. Feature Importance
```python
# Get feature importance
importance_df = results['feature_importance']

# Top 3 most important features
top_features = importance_df.head(3)
print(top_features)
```

### 3. Statistical Tests
```python
# Access hypothesis test results
hypothesis_results = results['hypothesis_tests']

for test in hypothesis_results:
    print(f"{test['name']}: {test['conclusion']}")
    print(f"  Statistic: {test['statistic']:.4f}")
    print(f"  P-value: {test['p_value']:.4e}")
```

## Batch Processing

### Processing Multiple Datasets
```python
datasets = [
    "data/netflix_reviews_jan.csv",
    "data/netflix_reviews_feb.csv",
    "data/netflix_reviews_mar.csv"
]

for dataset_path in datasets:
    print(f"Processing {dataset_path}...")
    
    analyzer = NetflixAnalyzer(dataset_path)
    results = analyzer.run_analysis()
    
    # Save results with unique names
    output_path = f"output/results_{os.path.basename(dataset_path)}"
    analyzer.save_preprocessed_data(output_path)
    
    print(f"Completed {dataset_path}")
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
import os

def process_dataset(dataset_path):
    analyzer = NetflixAnalyzer(dataset_path)
    results = analyzer.run_analysis()
    
    output_path = f"output/results_{os.path.basename(dataset_path)}"
    analyzer.save_preprocessed_data(output_path)
    
    return dataset_path, results['model_metrics']['accuracy']

# Process datasets in parallel
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_dataset, path) for path in datasets]
    
    for future in futures:
        dataset_path, accuracy = future.result()
        print(f"{dataset_path}: Accuracy = {accuracy:.4f}")
```

## Integration with Other Systems

### 1. Database Integration
```python
# Save results to database
import sqlite3

conn = sqlite3.connect('output/analytics.db')
results_df = analyzer.get_results_dataframe()
results_df.to_sql('churn_predictions', conn, if_exists='replace', index=False)
```

### 2. API Integration
```python
# Create REST API endpoint
from flask import Flask, request, jsonify

app = Flask(__name__)
analyzer = NetflixAnalyzer("data/default_reviews.csv")

@app.route('/predict', methods=['POST'])
def predict_churn():
    data = request.json
    content = data.get('review_content', '')
    score = data.get('rating', 5)
    
    # Create temporary DataFrame
    temp_df = pd.DataFrame([{
        'content': content,
        'score': score,
        'thumbsUpCount': 0
    }])
    
    # Process and predict
    processed = analyzer.processor.create_features(temp_df)
    prediction = analyzer.trainer.model.predict(processed[['sentiment_score', 'satisfaction_score', 'engagement_score']])
    
    return jsonify({'churn_risk': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

## Monitoring and Logging

### Enable Detailed Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/analysis.log'),
        logging.StreamHandler()
    ]
)

# Run analysis with logging
analyzer = NetflixAnalyzer("data/reviews.csv")
results = analyzer.run_analysis(verbose=True)
```

### Performance Monitoring
```python
import time

start_time = time.time()
results = analyzer.run_analysis()
end_time = time.time()

print(f"Analysis completed in {end_time - start_time:.2f} seconds")
print(f"Processed {len(analyzer.processed_df)} reviews")
```

## Troubleshooting Common Issues

### 1. Memory Issues
```python
# Process large datasets in chunks
analyzer = NetflixAnalyzer("data/large_reviews.csv")
analyzer.set_chunk_size(10000)  # Process 10K rows at a time
results = analyzer.run_analysis()
```

### 2. API Rate Limiting
```python
# Configure retry settings for API calls
from src.dspy_integration import DSPySentimentAnalyzer

analyzer = DSPySentimentAnalyzer()
analyzer.set_retry_config(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0
)
```

### 3. Model Performance Issues
```python
# Tune model hyperparameters
trainer = ModelTrainer(
    n_estimators=50,      # Reduce for faster training
    max_depth=10,         # Prevent overfitting
    class_weight='balanced'  # Handle imbalanced data
)
```

## Exporting Results

### 1. Export to CSV
```python
# Already handled by save_preprocessed_data()
analyzer.save_preprocessed_data("output/my_results.csv")
```

### 2. Export to JSON
```python
import json

# Export results as JSON
with open("output/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

### 3. Export to Database
```python
# Export to SQLite
import sqlite3

conn = sqlite3.connect('output/results.db')
df = analyzer.get_results_dataframe()
df.to_sql('predictions', conn, if_exists='replace', index=False)
```

## Advanced Usage Patterns

### 1. Incremental Learning
```python
# Update model with new data
new_data = pd.read_csv("data/new_reviews.csv")
analyzer.update_model(new_data)
```

### 2. A/B Testing
```python
# Compare different approaches
approaches = {
    'basic': {'use_advanced_nlp': False, 'use_advanced_sentiment': False},
    'advanced_nlp': {'use_advanced_nlp': True, 'use_advanced_sentiment': False},
    'foundation_models': {'use_advanced_nlp': False, 'use_advanced_sentiment': False, 'use_dspy_sentiment': True}
}

for name, params in approaches.items():
    print(f"Running {name} approach...")
    results = analyzer.run_analysis(**params)
    print(f"{name} accuracy: {results['model_metrics']['accuracy']:.4f}")
```

### 3. Custom Feature Engineering
```python
# Add custom features
def add_custom_features(df):
    df['review_length_category'] = pd.cut(df['review_length'], bins=[0, 50, 100, 200, float('inf')], 
                                         labels=['short', 'medium', 'long', 'very_long'])
    return df

analyzer.add_feature_engineering_step(add_custom_features)
results = analyzer.run_analysis()
```

## Performance Tips

### 1. Speed Optimization
```python
# Disable unnecessary features for faster processing
results = analyzer.run_analysis(
    use_advanced_nlp=False,      # Skip lemmatization
    use_advanced_sentiment=False # Use faster TextBlob
)
```

### 2. Memory Optimization
```python
# Process only necessary columns
analyzer = NetflixAnalyzer(
    "data/reviews.csv",
    columns_to_process=['content', 'score', 'thumbsUpCount']  # Only process these columns
)
```

### 3. Resource Management
```python
# Monitor resource usage
import psutil

process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB

results = analyzer.run_analysis()

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_after - memory_before:.2f} MB")
```

## Next Steps

1. **Explore Results**: Review output files and visualizations
2. **Customize Analysis**: Adjust parameters for your specific needs
3. **Integrate Systems**: Connect with your existing data infrastructure
4. **Monitor Performance**: Track model performance over time
5. **Share Insights**: Export results for stakeholder presentations

## Support

For usage issues:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed error messages
4. Include your system information and data sample