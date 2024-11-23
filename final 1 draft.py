import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class NetflixAnalyzer:
    def __init__(self, file_path):
        """Initialize with the path to the Netflix reviews CSV file."""
        self.df = pd.read_csv("C:/Users/Hrithik/Desktop/30 Days of Python/cleaned_netflix_reviews.csv")
        self.processed_df = None
        self.model = None
        self.pca = None
        self.scaler = None
        print(f"Loaded dataset with shape: {self.df.shape}")

    def preprocess_data(self):
        """Enhanced data preprocessing with text cleaning."""
        print("\n=== Data Preprocessing ===")
        
        # Initial data cleaning
        self.processed_df = self.df.copy()
        self.processed_df.drop_duplicates(subset='reviewId', inplace=True)
        self.processed_df.dropna(subset=['content'], inplace=True)
        
        # Text cleaning
        print("\nCleaning text data...")
        self.processed_df['cleaned_content'] = self.processed_df['content'].apply(self._clean_text)
        
        # Feature engineering
        print("Creating features...")
        self._create_features()
        
        print(f"\nFinal shape after preprocessing: {self.processed_df.shape}")
        
        # Display sample of processed data
        print("\nSample of processed features:")
        print(self.processed_df[['sentiment_score', 'satisfaction_score', 'engagement_score']].describe())
        
        return self.processed_df

    def _clean_text(self, text):
        """text cleaning with NLTK."""
        if not isinstance(text, str):
            return ''
            
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Join tokens back into text
        return ' '.join(tokens)

    def _create_features(self):
        """Create features for analysis."""
        df = self.processed_df
        
        # Text-based features
        df['review_length'] = df['cleaned_content'].str.len()
        df['word_count'] = df['cleaned_content'].apply(lambda x: len(str(x).split()))
        df['avg_word_length'] = df['cleaned_content'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()] or [0])
        )
        
        # Sentiment features
        df['sentiment_score'] = df['cleaned_content'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        df['subjectivity_score'] = df['cleaned_content'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        
        # Engagement and satisfaction scores
        df['engagement_score'] = (
            np.log1p(df['thumbsUpCount']) * 0.7 +
            df['word_count'] / df['word_count'].max() * 0.3
        ) * 100
        
        df['satisfaction_score'] = (
            df['score'] * 0.4 +
            (df['sentiment_score'] + 1) * 30 +
            df['engagement_score'] * 0.3
        ).clip(0, 100)
        
        # Churn risk calculation
        df['churn_risk'] = (
            (df['satisfaction_score'] < df['satisfaction_score'].quantile(0.3)) |
            (df['score'] <= 2) |
            (df['sentiment_score'] < -0.3)
        ).astype(int)

    def save_preprocessed_data(self, output_path):
        """Save preprocessed data with new columns to a CSV file."""
        if self.processed_df is not None:
            self.processed_df.to_csv(output_path, index=False)
            print(f"Preprocessed data saved to {output_path}")
        else:
            print("Data has not been preprocessed. Please run preprocess_data() first.")

    def perform_clustering(self):
        """Perform clustering with PCA."""
        print("\n=== Clustering Analysis ===")
        
        # Select features for clustering
        features = [
            'satisfaction_score', 'sentiment_score', 'engagement_score',
            'review_length', 'word_count', 'score'
        ]
        X = self.processed_df[features].values
        
        # Standardize and perform PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of components
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate explained variance ratio
        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance >= 0.95) + 1
        
        print(f"\nOptimal number of PCA components: {n_components}")
        print("Explained variance ratio:", explained_variance[:n_components])
        
        # Perform clustering on reduced dimensions
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.processed_df['cluster'] = kmeans.fit_predict(X_pca)
        
        # Analyze clusters
        cluster_stats = self.processed_df.groupby('cluster')[features].mean()
        print("\nCluster Profiles:")
        print(cluster_stats)
        
        # Visualize clusters
        self._plot_clusters(X_pca)

    def _plot_clusters(self, X_pca):
        """Create cluster visualization."""
        plt.figure(figsize=(12, 5))
        
        # Plot 1: First two PCA components
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=self.processed_df['cluster'],
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title('Clusters in PCA Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Plot 2: Cluster characteristics
        plt.subplot(1, 2, 2)
        cluster_sizes = self.processed_df['cluster'].value_counts()
        plt.pie(
            cluster_sizes,
            labels=[f'Cluster {i}\n({size} reviews)' for i, size in cluster_sizes.items()],
            autopct='%1.1f%%'
        )
        plt.title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.show()

    def perform_hypothesis_tests(self):
        """Conduct hypothesis testing."""
        print("\n=== Hypothesis Testing ===")
        
        tests = [
            self._test_satisfaction_churn(),
            self._test_engagement_sentiment(),
            self._test_cluster_churn(),
            self._test_length_satisfaction()
        ]
        
        for test in tests:
            print(f"\n{test['name']}:")
            print(f"Statistic: {test['statistic']:.4f}")
            print(f"P-value: {test['p_value']:.4e}")
            print(f"Conclusion: {test['conclusion']}")

    def _test_satisfaction_churn(self):
        """Test relationship between satisfaction and churn."""
        churned = self.processed_df[self.processed_df['churn_risk'] == 1]['satisfaction_score']
        retained = self.processed_df[self.processed_df['churn_risk'] == 0]['satisfaction_score']
        stat, p_value = ttest_ind(churned, retained)
        
        return {
            'name': 'H1: Satisfaction Score Impact on Churn',
            'statistic': stat,
            'p_value': p_value,
            'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }

    def _test_engagement_sentiment(self):
        """Test correlation between engagement and sentiment."""
        correlation, p_value = spearmanr(
            self.processed_df['engagement_score'],
            self.processed_df['sentiment_score']
        )
        
        return {
            'name': 'H2: Engagement-Sentiment Correlation',
            'statistic': correlation,
            'p_value': p_value,
            'conclusion': 'Significant correlation' if p_value < 0.05 else 'No significant correlation'
        }

    def _test_cluster_churn(self):
        """Test association between clusters and churn risk."""
        contingency = pd.crosstab(self.processed_df['cluster'], self.processed_df['churn_risk'])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        return {
            'name': 'H3: Cluster-Churn Association',
            'statistic': chi2,
            'p_value': p_value,
            'conclusion': 'Significant association' if p_value < 0.05 else 'No significant association'
        }


    def _test_length_satisfaction(self):
        """Test correlation between review length and satisfaction."""
        correlation, p_value = spearmanr(
            self.processed_df['review_length'],
            self.processed_df['satisfaction_score']
        )
        
        return {
            'name': 'H4: Review Length-Satisfaction Correlation',
            'statistic': correlation,
            'p_value': p_value,
            'conclusion': 'Significant correlation' if p_value < 0.05 else 'No significant correlation'
        }

    def train_model(self):
        """Train an optimized predictive model."""
        print("\n===  Predictive Modeling ===")
        
        # Prepare features
        features = [
            'satisfaction_score', 'sentiment_score', 'engagement_score',
            'review_length', 'word_count', 'score', 'subjectivity_score'
        ]
        X = self.processed_df[features]
        y = self.processed_df['churn_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train optimized random forest
        self.model = RandomForestClassifier(
            n_estimators=35,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model and evaluate
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Plot ROC curve
        self._plot_roc_curve(X_test, y_test)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)

    def _plot_roc_curve(self, X_test, y_test):
        """Plot ROC curve for model evaluation."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

def main():
    # Initialize analyzer
    analyzer = NetflixAnalyzer("C:/Users/Hrithik/Desktop/30 Days of Python/cleaned_netflix_reviews.csv")
    
    # Run analysis pipeline
    analyzer.preprocess_data()
    analyzer.save_preprocessed_data("C:/Users/Hrithik/Desktop/30 Days of Python/processed_netflix_reviews.csv")
    analyzer.perform_clustering()
    analyzer.perform_hypothesis_tests()
    analyzer.train_model()

if __name__ == "__main__":
    main()

print(df1.head())
print(df1.columns)

df1 = pd.read_csv("C:/Users/Hrithik/Desktop/30 Days of Python/processed_netflix_reviews.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and prepare data
df1 = pd.read_csv("C:/Users/Hrithik/Desktop/30 Days of Python/processed_netflix_reviews.csv")
features = ['satisfaction_score', 'sentiment_score', 'engagement_score']
X = df1[features].dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Gaussian Mixture Model clustering with different covariance type
gmm = GaussianMixture(n_components=4, random_state=42, covariance_type = 'tied')
gmm_labels = gmm.fit_predict(X_scaled)

# Plot as before but now with the adjusted GMM model
plt.figure(figsize=(14, 6))

# K-means Plot
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=X['satisfaction_score'], y=X['sentiment_score'],
    hue=kmeans_labels, palette='viridis', style=kmeans_labels, legend='full'
)
plt.title('K-means Clustering')
plt.xlabel('Satisfaction Score')
plt.ylabel('Sentiment Score')

# GMM Plot with 'tied' covariance type
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=X['satisfaction_score'], y=X['sentiment_score'],
    hue=gmm_labels, palette='viridis', style=gmm_labels, legend='full'
)
plt.title('Gaussian Mixture Model (GMM) Clustering')
plt.xlabel('Satisfaction Score')
plt.ylabel('Sentiment Score')

plt.tight_layout()
plt.show()
