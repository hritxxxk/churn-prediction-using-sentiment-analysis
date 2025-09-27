#!/bin/bash

# Script to push the clean version to your GitHub repository

echo "=== Netflix Churn Prediction System - GitHub Update Script ==="
echo ""

# Check if we're in the right directory
if [ ! -d "clean_version" ]; then
    echo "Error: clean_version directory not found!"
    echo "Please run this script from the churnprediction directory."
    exit 1
fi

echo "This script will help you update your GitHub repository with the clean version."
echo ""
echo "IMPORTANT: This will replace all files in your current repository."
echo "Make sure you have backed up any important changes before proceeding."
echo ""

read -p "Do you want to continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "=== Step 1: Checking current Git status ==="
if [ ! -d ".git" ]; then
    echo "Error: No Git repository found in current directory!"
    echo "Please make sure you're in your repository directory."
    exit 1
fi

echo "Current branch: $(git branch --show-current)"
echo "Remote origin: $(git remote get-url origin)"
echo ""

echo "=== Step 2: Creating backup branch ==="
git checkout -b backup-before-clean-update-$(date +%Y%m%d-%H%M%S) 2>/dev/null || {
    echo "Creating backup branch..."
    git checkout -b backup-before-clean-update
}
git add .
git commit -m "Backup before clean version update" --allow-empty
git push origin backup-before-clean-update || echo "Could not push backup branch to origin"
echo ""

echo "=== Step 3: Switching to main branch ==="
git checkout main 2>/dev/null || git checkout master 2>/dev/null || {
    echo "Could not switch to main/master branch"
    echo "Available branches:"
    git branch
    read -p "Enter branch name to use: " branch_name
    git checkout $branch_name
}
echo ""

echo "=== Step 4: Preparing clean version ==="
# Create temporary directory for clean files
TEMP_DIR="/tmp/netflix_clean_$(date +%s)"
mkdir -p "$TEMP_DIR"

# Copy clean version files
cp -r clean_version/* "$TEMP_DIR/"

echo "=== Step 5: Updating repository ==="
# Remove all files except .git directory
find . -mindepth 1 -maxdepth 1 ! -name '.git' ! -name 'PUSH_TO_GITHUB.sh' ! -name 'SUMMARY_AND_INSTRUCTIONS.md' -exec rm -rf {} +

# Copy clean files
cp -r "$TEMP_DIR"/* .

# Clean up
rm -rf "$TEMP_DIR"

echo "=== Step 6: Committing changes ==="
git add .
git status

echo ""
echo "Commit message preview:"
echo "feat: Replace with enhanced Netflix churn prediction system"
echo ""
echo "- Fixed critical data leakage issue that was causing 99.9% 'accuracy'"
echo "- Now achieves realistic 76.2% accuracy with proper evaluation"
echo "- Enhanced with advanced NLP and sentiment analysis techniques"
echo "- Added comprehensive documentation and professional code structure"
echo "- Implemented proper security practices with no hardcoded API keys"
echo "- Added DSPy foundation model integration for enhanced capabilities"
echo ""

read -p "Do you want to commit these changes? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Changes prepared but not committed. You can review and commit manually."
    exit 0
fi

git commit -m "feat: Replace with enhanced Netflix churn prediction system

- Fixed critical data leakage issue that was causing 99.9% 'accuracy'
- Now achieves realistic 76.2% accuracy with proper evaluation
- Enhanced with advanced NLP and sentiment analysis techniques
- Added comprehensive documentation and professional code structure
- Implemented proper security practices with no hardcoded API keys
- Added DSPy foundation model integration for enhanced capabilities"

echo ""
echo "=== Step 7: Pushing to GitHub ==="
read -p "Do you want to push to GitHub now? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Changes committed locally. You can push manually with 'git push origin main'"
    exit 0
fi

git push origin main || git push origin master || {
    echo "Could not push to main or master. Please push manually."
    echo "Available remote branches:"
    git branch -r
}

echo ""
echo "=== Update Complete! ==="
echo ""
echo "Your GitHub repository has been updated with the clean version."
echo "Please verify on GitHub that all files are present and no sensitive information is exposed."
echo ""
echo "Next steps:"
echo "1. Visit your repository on GitHub to verify the update"
echo "2. Update your README and other documentation as needed"
echo "3. Add your actual data files (they're excluded by .gitignore)"
echo "4. Configure API keys in .env for enhanced capabilities (optional)"
echo ""
echo "For detailed instructions, see SUMMARY_AND_INSTRUCTIONS.md"