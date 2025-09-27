# Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 500MB free space for code and dependencies

### Recommended Requirements
- **Operating System**: Ubuntu 20.04+, macOS 12+, or Windows 11+
- **Python**: 3.9 or higher
- **RAM**: 16GB or more
- **Disk Space**: 2GB free space
- **GPU**: CUDA-compatible GPU (optional, for faster processing)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/hritxxxk/churn-prediction-using-sentiment-analysis.git
cd churn-prediction-using-sentiment-analysis
```

### 2. Set Up Python Environment

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Using Conda (Alternative)
```bash
# Create conda environment
conda create -n netflix-churn python=3.9

# Activate conda environment
conda activate netflix-churn
```

### 3. Install Dependencies

#### Install Core Dependencies
```bash
pip install -r requirements.txt
```

#### Install Development Dependencies (Optional)
```bash
pip install -e ".[dev]"
```

### 4. Download NLTK Data
The system requires NLTK data for text processing:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Or run the download script:
```bash
python -m src.utils.download_nltk_data
```

### 5. Set Up Environment Variables

#### Create .env File
```bash
cp .env.example .env
```

#### Edit .env File
Add your API keys to the `.env` file:
```bash
nano .env
```

Example content:
```
# Google API Key for Gemini models (optional)
GOOGLE_API_KEY=your_actual_google_api_key_here

# OpenAI API Key (optional)
OPENAI_API_KEY=your_actual_openai_api_key_here

# Anthropic API Key (optional)
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
```

### 6. Prepare Data

#### Data Structure
Place your Netflix review data in `data/processed_netflix_reviews.csv` with the following columns:
- `reviewId`
- `userName`
- `content`
- `score`
- `thumbsUpCount`
- `reviewCreatedVersion`
- `at`
- `appVersion`

#### Sample Data
A small sample dataset is included for testing purposes in `data/small_sample.csv`.

## Testing the Installation

### Run Basic Tests
```bash
# Run unit tests
python -m pytest tests/

# Run a quick demo
python -m src.main --data data/small_sample.csv
```

### Verify Installation
```python
from src.analysis.analyzer import NetflixAnalyzer

# Initialize with sample data
analyzer = NetflixAnalyzer("data/small_sample.csv")

# Run quick analysis
results = analyzer.run_analysis()

print("Installation verified successfully!")
```

## Common Installation Issues

### 1. Python Version Issues
**Problem**: "Python 3.8 or higher required"
**Solution**: Upgrade Python or use pyenv to manage versions
```bash
# Using pyenv
pyenv install 3.9.16
pyenv local 3.9.16
```

### 2. Dependency Installation Failures
**Problem**: "Failed building wheel for ..." or similar errors
**Solution**: Install system dependencies first
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Microsoft C++ Build Tools
```

### 3. NLTK Download Issues
**Problem**: "Resource punkt not found"
**Solution**: Manually download NLTK data
```python
import nltk
nltk.download('all')  # Download all NLTK data
```

### 4. TensorFlow/PyTorch Issues
**Problem**: "No module named tensorflow" or "No module named torch"
**Solution**: Install specific versions
```bash
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 5. Permission Errors
**Problem**: "Permission denied" during installation
**Solution**: Use virtual environment or install with --user flag
```bash
pip install --user -r requirements.txt
```

## Advanced Installation Options

### 1. Docker Installation (Coming Soon)
```bash
# Build Docker image
docker build -t netflix-churn .

# Run container
docker run -v $(pwd)/data:/app/data netflix-churn
```

### 2. GPU Acceleration
For faster processing with compatible GPUs:
```bash
# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

### 3. Production Deployment
For production environments:
```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run as service
gunicorn --bind 0.0.0.0:8000 src.app:app
```

## Performance Optimization

### 1. Memory Management
```bash
# Set memory limits for large datasets
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2
```

### 2. Parallel Processing
```bash
# Use multiple cores for data processing
export OMP_NUM_THREADS=4
```

### 3. Caching
Enable caching for repeated analyses:
```python
from src.utils.cache import enable_caching
enable_caching()
```

## Verification Checklist

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment activated (recommended)
- [ ] All dependencies installed successfully
- [ ] NLTK data downloaded
- [ ] Environment variables configured (optional)
- [ ] Sample data available
- [ ] Basic tests pass
- [ ] Quick demo runs successfully

## Next Steps

1. **Prepare Your Data**: Follow the data preparation guide
2. **Configure API Keys**: Set up foundation model access (optional)
3. **Run Full Analysis**: Execute the complete pipeline
4. **Customize Parameters**: Adjust settings for your specific use case
5. **Deploy**: Set up for production use (if applicable)

## Support

For installation issues, please:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed error messages
4. Include your system information (OS, Python version, etc.)