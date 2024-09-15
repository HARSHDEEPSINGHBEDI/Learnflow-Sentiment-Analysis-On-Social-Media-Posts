# Learnflow Sentiment Analysis on Social Media Posts

## Project Overview

This project focuses on sentiment analysis of Instagram reviews using natural language processing (NLP) techniques and machine learning models. The goal is to classify sentiments into various categories using polarity scores derived from the review texts.

## Key Accomplishments

1. **Imported Essential Libraries**: 
   - Libraries such as NumPy, Pandas, Matplotlib, Seaborn, NLTK, WordCloud, and scikit-learn were used for data manipulation, visualization, and modeling.

2. **Loaded Dataset**: 
   - A dataset containing Instagram reviews was loaded from a CSV file for analysis.
     ```python
     url = 'https://learnflow.cloud/loginpage/dashboard/source%20code/2_week_program/ML_2_week/sentimentdataset.csv'
     ```

3. **Explored the Data**: 
   - Conducted data exploration to understand its dimensions, summary statistics, missing values, and duplicates.

4. **Data Cleaning**: 
   - Removed the `'Unnamed: 0.1', 'Unnamed: 0', 'Timestamp', 'User', 'Hashtags'` columns, eliminated duplicates, and cleaned `Text` by removing mentions, hashtags, retweets, and URLs.

5. **Performed Sentiment Analysis**: 
   - Utilized TextBlob to calculate sentiment polarity and subjectivity for each review.

6. **Generated Word Cloud**: 
   - Created a word cloud to visualize frequently occurring words in the reviews.

7. **Categorized Sentiments**: 
   - Defined a function to categorize sentiments (positive, negative, neutral) based on polarity scores, and applied it to create a new `Sentiment_Analysis` column.

8. **Visualized Data**: 
   - Used KDE plots and histograms to visualize the distribution of polarity and sentiment categories.

9. **Implemented Machine Learning**: 
   - Created a pipeline with vectorization, TF-IDF transformation, and a Multinomial Naive Bayes classifier to predict sentiment categories.

10. **Trained and Evaluated Model**: 
    - Trained the model on a training set, made predictions on a test set, and evaluated performance using metrics such as accuracy, confusion matrix, and classification report.

## Approaches Used

### Approach 1: Using Pre-defined Sentiments

This approach involved using the default 'Sentiment' column provided in the dataset to train the model and obtain accuracy. However, this method does not align with the goal of deriving sentiments directly from the text.

### Approach 2: Using Polarity to Derive Sentiments

This approach involved the following steps:

- **Text Processing**: Used libraries like TextBlob to calculate the polarity of each review's text.
  
  #### A) Fine-Grained Sentiment Categorization
  - **Custom Sentiment Categories**:
    - Created a function (`categorize_sentiment`) to assign custom sentiment categories based on polarity scores.
    - This approach categorized emotions into a fine-grained list of unique sentiments, leading to **46-48% accuracy** in RandomForest, SVM, and MNB models.

    ```python
    # Unique Sentiments
    sentiment_list = [
        # Positive Emotions
        'Euphoria', 'Ecstasy', 'Elation', 'Joy', 'Excitement', 'Happiness', 'Love',
        'Contentment', 'Amusement', 'Admiration', 'Affection', 'Awe', 'Adoration', 
        'Calmness', 'Surprise', 'Anticipation', 'Pride', 'Gratitude', 'Playful', 'Hopeful',
        
        # Neutral Emotion
        'Neutral',
        
        # Mild Negative Emotions
        'Boredom', 'Confusion', 'Sadness', 'Fear', 'Anger', 'Disgust', 'Shame', 
        'Frustration', 'Guilt', 'Melancholy',
        
        # Strong Negative Emotions
        'Grief', 'Despair', 'Loneliness', 'Betrayal', 'Desolation', 'Severe Sadness',
        
        # Undefined Category
        'Undefined Emotion'
    ]
    ```

  #### B) Broader Sentiment Categorization
  - **Custom Sentiment Categories**:
    - Created a function (`map_sentiment_category`) to map sentiments into broader categories based on polarity scores.
    - This approach yielded **65-70% accuracy** in RandomForest, SVM, and MNB models.

    ```python
    # Broader Sentiment Categories
    
    # Positive Sentiments
    positive_sentiments = [
        'Positive', 'Happiness', 'Joy', 'Love', 'Excitement', 'Enjoyment', 'Elation', 'Euphoria',
        'Admiration', 'Affection', 'Awe', 'Contentment', 'Serenity', 'Gratitude', 'Hope',
        'Empowerment', 'Compassion', 'Tenderness', 'Amusement', 'Enthusiasm', 'Fulfillment', 
        'Pride', 'Surprise', 'Adoration', 'Optimism', 'Confidence'
    ]

    # Neutral Sentiments
    neutral_sentiments = [
        'Neutral', 'Calmness', 'Confusion', 'Acceptance', 'Indifference', 'Ambivalence', 
        'Serenity', 'Curiosity', 'Reflection', 'Contemplation', 'Tranquility'
    ]

    # Negative Sentiments
    negative_sentiments = [
        'Anger', 'Fear', 'Sadness', 'Disgust', 'Bitter', 'Resentment', 'Jealousy', 'Regret', 
        'Frustration', 'Loneliness', 'Despair', 'Grief', 'Anxiety', 'Helplessness', 'Desperation', 
        'Isolation', 'Heartbreak', 'Betrayal', 'Sorrow', 'Melancholy'
    ]
    ```

## Conclusion

The broader sentiment categorization approach provided better accuracy compared to the fine-grained classification. Although finer sentiment categories offer more detailed emotional insight, they are more challenging to classify accurately due to overlapping categories and similar textual patterns.

## Future Work

- Explore advanced models like LSTM and BERT for improved accuracy if given opportunity.



