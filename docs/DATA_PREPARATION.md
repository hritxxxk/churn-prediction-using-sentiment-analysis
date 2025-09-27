# Data Preparation Guide

## Overview
This guide explains how to prepare data for the Netflix Churn Prediction System. The system requires Netflix App Store review data in a specific format to function properly.

## Required Data Format

### Input File Structure
The system expects a CSV file with the following columns:

| Column Name | Description | Data Type | Example |
|-------------|-------------|-----------|---------|
| reviewId | Unique identifier for the review | String | "cb8b1fa9-10ab-4712-9257-a1e39a11db17" |
| userName | Name of the reviewer | String | "John Doe" |
| content | Text content of the review | String | "Great series with amazing graphics!" |
| score | Star rating (1-5) | Integer | 5 |
| thumbsUpCount | Number of helpful votes | Integer | 10 |
| reviewCreatedVersion | App version when review was created | String | "8.135.1 build 7 50902" |
| at | Timestamp of review creation | Datetime | "2024-10-11 14:37:28" |
| appVersion | Current app version | String | "8.135.1 build 7 50902" |

### Sample Data Row
```csv
reviewId,userName,content,score,thumbsUpCount,reviewCreatedVersion,at,appVersion
"cb8b1fa9-10ab-4712-9257-a1e39a11db17","John Doe","Great series with amazing graphics!",5,10,"8.135.1 build 7 50902","2024-10-11 14:37:28","8.135.1 build 7 50902"
```

## Data Collection Methods

### 1. Web Scraping (Manual)
If you need to collect your own data:

#### Using Google Play Store Scraper
```bash
# Install scraper
pip install google-play-scraper

# Collect Netflix reviews
python -m src.utils.scraper --package com.netflix.mediaclient --limit 1000
```

#### Using App Store Scraper
```bash
# Install scraper
pip install app-store-scraper

# Collect Netflix reviews
python -m src.utils.scraper --country US --limit 1000
```

### 2. Using Provided Data
The repository includes a sample dataset:
- Location: `data/small_sample.csv`
- Size: 100 reviews
- Purpose: Testing and demonstration

## Data Preprocessing Pipeline

### 1. Loading Data
```python
from src.data_processing.data_processor import DataProcessor

processor = DataProcessor()
df = processor.load_data("path/to/your/data.csv")
```

### 2. Cleaning Data
```python
df_cleaned = processor.clean_data(df)
```

### 3. Feature Engineering
```python
df_features = processor.create_features(df_cleaned)
```

### 4. Text Preprocessing
```python
df_processed = df_features.copy()
df_processed['cleaned_content'] = df_features['content'].apply(processor.clean_text)
```

## Required Columns for Analysis

### Essential Columns
These columns are absolutely required for the system to function:

1. `content` - Review text (String)
2. `score` - Star rating 1-5 (Integer)
3. `thumbsUpCount` - Helpful votes (Integer)

### Recommended Columns
These columns enhance analysis quality:

1. `reviewId` - Unique identifier (String)
2. `userName` - Reviewer name (String)
3. `reviewCreatedVersion` - App version (String)
4. `at` - Review timestamp (Datetime)
5. `appVersion` - Current version (String)

## Data Quality Guidelines

### 1. Completeness
- Minimize missing values in essential columns
- Handle missing data appropriately:
  ```python
  # Fill missing scores with median
  df['score'].fillna(df['score'].median(), inplace=True)
  
  # Remove rows with missing content
  df.dropna(subset=['content'], inplace=True)
  ```

### 2. Consistency
- Ensure consistent data types across columns
- Standardize text encoding (UTF-8 recommended)
- Normalize timestamps to consistent format

### 3. Accuracy
- Validate score range (1-5)
- Check for realistic thumbsUpCount values
- Verify date formats

## Data Validation

### Automated Validation Script
```python
from src.utils.validator import validate_data

# Validate your data
is_valid, errors = validate_data("path/to/your/data.csv")

if is_valid:
    print("Data is valid for analysis")
else:
    print("Data validation errors:")
    for error in errors:
        print(f"- {error}")
```

### Manual Validation Checklist
- [ ] CSV format with proper headers
- [ ] UTF-8 encoding
- [ ] Essential columns present
- [ ] Score between 1-5
- [ ] ThumbsUpCount ≥ 0
- [ ] No completely empty rows
- [ ] Reasonable data size (not all zeros or identical)

## Sample Data Generation

### Generating Test Data
```python
from src.utils.data_generator import generate_sample_data

# Generate 1000 sample reviews
sample_df = generate_sample_data(1000)
sample_df.to_csv("data/generated_sample.csv", index=False)
```

### Using Sample Data
```python
# Load sample data for testing
analyzer = NetflixAnalyzer("data/small_sample.csv")
results = analyzer.run_analysis()
```

## Large Dataset Considerations

### Memory Management
For datasets > 100,000 reviews:

1. **Chunk Processing**:
   ```python
   # Process data in chunks
   chunk_size = 10000
   for chunk in pd.read_csv("large_dataset.csv", chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **Sampling**:
   ```python
   # Use random sample for analysis
   df_sample = df.sample(n=50000, random_state=42)
   ```

### Performance Optimization
1. Use SSD storage for data files
2. Allocate sufficient RAM (16GB+ recommended)
3. Close unnecessary applications during processing
4. Use multiprocessing for parallel processing:
   ```python
   from src.utils.parallel_processor import process_parallel
   
   results = process_parallel(df, num_processes=4)
   ```

## Data Privacy and Ethics

### Anonymization
When working with real user data:
```python
# Remove personally identifiable information
df['userName'] = df['userName'].apply(lambda x: hash(x) if pd.notnull(x) else x)
```

### Compliance
- Follow GDPR, CCPA, and other privacy regulations
- Obtain proper permissions for data usage
- Implement data retention policies
- Use secure data storage practices

## Troubleshooting Data Issues

### Common Problems and Solutions

1. **Encoding Issues**
   ```python
   # Specify encoding when loading
   df = pd.read_csv("data.csv", encoding='utf-8')
   ```

2. **Memory Errors**
   ```python
   # Use chunking for large files
   df = pd.read_csv("data.csv", chunksize=10000)
   ```

3. **Missing Columns**
   ```python
   # Add missing columns with default values
   if 'thumbsUpCount' not in df.columns:
       df['thumbsUpCount'] = 0
   ```

4. **Incorrect Data Types**
   ```python
   # Convert columns to correct types
   df['score'] = pd.to_numeric(df['score'], errors='coerce')
   df['thumbsUpCount'] = pd.to_numeric(df['thumbsUpCount'], errors='coerce').fillna(0).astype(int)
   ```

## Data Transformation Examples

### Converting Different Formats
```python
# JSON to CSV
import json
with open('reviews.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.to_csv('reviews.csv', index=False)

# Excel to CSV
df = pd.read_excel('reviews.xlsx')
df.to_csv('reviews.csv', index=False)
```

### Adding Missing Columns
```python
# Add dummy columns if missing
required_columns = ['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'appVersion']

for col in required_columns:
    if col not in df.columns:
        if col in ['score', 'thumbsUpCount']:
            df[col] = 0
        else:
            df[col] = ''

# Generate unique IDs if missing
if 'reviewId' not in df.columns or df['reviewId'].isnull().all():
    df['reviewId'] = [str(uuid.uuid4()) for _ in range(len(df))]
```

## Validation Script

### Automated Data Validation
```python
def validate_netflix_data(df):
    """Validate Netflix review data format."""
    required_columns = ['content', 'score', 'thumbsUpCount']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check score range
    invalid_scores = df[(df['score'] < 1) | (df['score'] > 5)]
    if not invalid_scores.empty:
        return False, f"Invalid scores found: {len(invalid_scores)} rows"
    
    # Check thumbsUpCount
    if (df['thumbsUpCount'] < 0).any():
        return False, "Negative thumbsUpCount values found"
    
    # Check content
    if df['content'].isnull().all():
        return False, "No content data found"
    
    return True, "Data is valid"

# Usage
is_valid, message = validate_netflix_data(df)
print(message)
```

## Next Steps

1. **Prepare Your Data**: Follow the format requirements above
2. **Validate Data Quality**: Use the validation tools
3. **Place Data in Correct Location**: Put your CSV file in the `data/` directory
4. **Run Analysis**: Execute the main analysis pipeline
5. **Review Results**: Check output files and visualizations

## Support

For data preparation issues:
1. Check the validation script output
2. Review the sample data format
3. Ensure all required columns are present
4. Contact support with specific error messages