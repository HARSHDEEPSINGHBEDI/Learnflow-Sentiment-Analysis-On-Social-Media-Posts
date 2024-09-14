# Learnflow-Sentiment-Analysis-On-Social-Media-Posts

Here is a concise breakdown of what I accomplished in the provided code:

1. **Imported Essential Libraries**: I started by importing crucial Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, NLTK, WordCloud, and scikit-learn.

2. **Loaded Dataset**: I loaded a dataset containing Instagram reviews from a CSV file.

3. **Explored the Data**: I conducted an initial exploration of the dataset, understanding its dimensions, summary statistics, basic information, as well as the presence of missing values and duplicates.

4. **Data Cleaning**: I proceeded to clean the data by removing the 'review_date' column, eliminating duplicated records, and sanitizing the 'review_description' text by eliminating mentions, hashtags, retweets, and hyperlinks.

5. **Performed Sentiment Analysis**: I employed the TextBlob library to compute sentiment subjectivity and polarity for each review description.

6. **Generated Word Cloud**: I created a word cloud visualization based on the processed review descriptions, offering a visual representation of common terms.

7. **Categorized Sentiments**: I defined a function to categorize sentiments (negative, neutral, positive) relying on polarity values. This categorization was then applied to produce a new 'Sentiment Analysis' column.

8. **Visualized Data**: I employed various visualizations, including KDE plots and histograms, to showcase the distribution of polarity and the count of different sentiment categories.

9. **Implemented Machine Learning**: I transformed sentiment labels into numerical values, prepped the data for model training, and constructed a pipeline incorporating vectorization, TF-IDF transformation, and a Multinomial Naive Bayes classifier.

10. **Trained and Evaluated Model**: I trained the pipeline on a designated training set, utilized it to make predictions on a separate test set, and gauged the model's performance via metrics like a confusion matrix, classification report, and accuracy score.

In summary, I conducted a comprehensive sentiment analysis of Instagram reviews, undertook data cleaning and preparation, visualized sentiment distributions, and crafted a machine learning model for sentiment classification.
