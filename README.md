# churn-prediction-using-sentiment-analysis
# Enhanced Netflix Analyzer for Sentiment-Driven Churn Prediction

This project focuses on building a comprehensive analytical framework for exploring customer sentiment and churn risk using data from Netflix reviews. The Python-based solution provides insights into user behavior, satisfaction levels, and churn risk, leveraging advanced data preprocessing, sentiment analysis, clustering, and predictive modeling.

#### Key Features and Workflow:
1. **Data Preprocessing**:
   - Loaded Netflix reviews dataset containing detailed customer feedback.
   - Conducted extensive text cleaning using regular expressions and NLTK, including stopword removal and tokenization.
   - Engineered multiple features such as review length, word count, average word length, sentiment score (polarity), and subjectivity score using TextBlob.
   - Derived engagement and satisfaction scores, combining sentiment, review metrics, and thumbs-up counts to quantify user engagement and satisfaction.
   - Identified potential churners using a composite metric based on satisfaction scores, low ratings, and negative sentiment.

2. **Data Insights and Storage**:
   - Processed data was saved with newly created features, allowing easy reuse and further analysis.
   - Summarized descriptive statistics to highlight trends in user satisfaction and engagement.

3. **Clustering Analysis**:
   - Applied **Principal Component Analysis (PCA)** to reduce dimensionality, selecting the optimal number of components that explain 95% variance.
   - Performed **k-means clustering** to segment users into distinct behavioral groups, analyzing average satisfaction, engagement, and churn risk across clusters.
   - Visualized clusters in reduced-dimensional PCA space and depicted the size distribution of each cluster for interpretability.

4. **Hypothesis Testing**:
   - Conducted four hypothesis tests, including:
     - Relationship between satisfaction and churn using t-tests.
     - Correlation between engagement and sentiment via Spearman correlation.
     - Association between clusters and churn risk using chi-square tests.
     - Relationship between review length and satisfaction through statistical correlation.
   - Provided statistical evidence to support actionable insights into churn behavior.

5. **Predictive Modeling**:
   - Developed a **Random Forest Classifier** with optimized hyperparameters to predict churn risk, achieving robust cross-validation scores.
   - Evaluated model performance through metrics like precision, recall, and ROC curves.
   - Conducted feature importance analysis, ranking the most influential factors driving churn risk.

6. **Clustering Refinement**:
   - Compared clustering outcomes from **k-means** and **Gaussian Mixture Models (GMM)**, incorporating tied covariance structures for better adaptability.
   - Presented visual comparisons of clustering effectiveness using satisfaction and sentiment scores.

#### Tools and Technologies:
- **Libraries**: pandas, numpy, matplotlib, seaborn, sklearn, TextBlob, NLTK, scipy.stats.
- **Key Techniques**: Text cleaning, feature engineering, PCA, clustering (k-means, GMM), Random Forest modeling, hypothesis testing.
- **Visualization**: Scatter plots, pie charts, ROC curves, and comparative clustering visualizations.

#### Outcomes:
- Identified significant factors influencing user churn, including sentiment scores, satisfaction levels, and engagement metrics.
- Clustered users into actionable groups, enabling targeted interventions to reduce churn.
- Demonstrated the effectiveness of combining text analytics, statistical hypothesis testing, and machine learning for customer insights.

#### Use Cases:
- Businesses seeking to reduce customer churn by identifying dissatisfaction patterns.
- Applications in understanding customer reviews and improving product features or services.
  
This project exemplifies a data-driven approach to solving real-world business problems, leveraging machine learning and statistical insights.
