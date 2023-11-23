## Sentiment Analysis of Ideas and Comments (1.1 innovation.py)

### Overview
In this project, we explore the emotions and opinions expressed in ideas and comments using sentiment analysis. The script is written in Python, leveraging libraries like `pandas`, `BeautifulSoup`, and `NLTK`.

### Key Steps

1. **Data Preparation**: 
   - Load ideas and comments from JSON files.
   - Convert HTML content in the descriptions to plain text for clarity.

2. **Sentiment Analysis**: 
   - Utilize `NLTK`'s VADER tool to analyze sentiments of ideas and comments.
   - Calculate a sentiment score for each idea and comment, indicating positive, neutral, or negative sentiment.

3. **Data Aggregation**: 
   - Group comments by their associated idea and calculate the average sentiment score for comments on each idea.

4. **Data Merging**: 
   - Combine the ideas and their average comment sentiment scores into one DataFrame for comprehensive analysis.

5. **Visualization**: 
   - Create a bar chart using `matplotlib` and `numpy` to compare the sentiment scores.
   - Display idea sentiments in blue and average comment sentiments in yellow for a clear visual comparison.

### Results
The resulting plot provides a visual representation of the sentiments associated with each idea and its comments. This analysis helps in understanding public opinion and emotions towards these ideas, offering valuable insights.

<img width="1706" alt="Screen Shot 2023-11-16 at 10 26 33 PM" src="https://github.com/joaovasco01/SentimentAnalysis_Ideas-Comments/assets/61276111/e15ac17a-deac-486d-a3da-99fdde806a6f">

## Own Interpretation

The graphical analysis suggests that the sentiment scores of idea descriptions are more variable compared to the average sentiment of the comments. This observation might stem from the tendency of idea descriptions to be more expressive or exaggerated in nature, resulting in a wider range of sentiment scores. This contrast highlights how different modes of expression (ideas vs. comments) can vary in emotional intensity and variability.


## Own Research PreTrainedPortugueseAnalysis.py
## Sentiment Analysis Results

### English Text Sentiment
- **Negative Sentiment Probability:** 0.0596
- **Neutral Sentiment Probability:** 0.8874
- **Positive Sentiment Probability:** 0.0529
- **Interpretation:** The sentiment of the English text is predominantly neutral.

### Portuguese Text Sentiment
- **Negative Sentiment Probability:** 0.0489
- **Neutral Sentiment Probability:** 0.9181
- **Positive Sentiment Probability:** 0.0329
- **Interpretation:** The sentiment of the Portuguese text is predominantly neutral.

For this analysis, I utilized the pretrained model from [pysentimiento/bertweet-pt-sentiment](https://huggingface.co/pysentimiento/bertweet-pt-sentiment) ([Research Paper](https://arxiv.org/abs/2106.09462)). This decision was motivated by the need to ensure accuracy in sentiment analysis of Portuguese texts. My objective was to verify if the existing sentiment analysis tool was appropriate for Portuguese descriptions. The results from the pretrained model confirmed that the previous sentiment analysis was accurate, affirming its suitability for our project. It was done to re-assure the results I had previously on the sentiment of each description, since the results were pretty similar there is no need for changes on innovation.py.




### 1.2 Currently being updated

# Part II - Data Analysis

## Overview

This exercise involves analyzing a dataset of Amazon product reviews. The dataset is in JSON format and includes various fields such as the reviewer's ID, product ID, review text, ratings, and votes for helpfulness.

## Objectives

The analysis aims to answer the following questions using the dataset:

1. **Is there a correlation between the product's rating and the review's helpfulness?**
2. **Who are the most helpful reviewers?**
3. **Have reviews been getting more or less helpful over time?**

## Data

The dataset is structured with the following columns:

- `reviewerID`: ID of the reviewer (e.g., A2SUAM1J3GNN3B)
- `asin`: ID of the product (e.g., 0000013714)
- `reviewerName`: Name of the reviewer
- `vote`: Helpful votes of the review
- `style`: A dictionary of the product metadata (e.g., format)
- `reviewText`: Text of the review
- `overall`: Rating of the product
- `summary`: Summary of the review
- `unixReviewTime`: Time of the review (unix time)
- `reviewTime`: Time of the review (raw)

## Analysis

### Code Implementation (`21.py`)

## Analysis Details

- **Correlation between Product Rating and Review Helpfulness:** The `corr` function in Pandas calculates the statistical relationship between product ratings (`overall`) and review helpfulness (`vote`). The result indicates the nature of this relationship.
- **Most Helpful Reviewers:** Reviewers are grouped by their ID, and their votes are summed to identify the most helpful ones.
- **Trend of Review Helpfulness Over Time:** The dataset is grouped by year, and the average helpfulness votes are calculated to observe trends over time.

## Results

### Correlation between Product Rating and Review Helpfulness

- **Output:** `Correlation: -0.017070877080897048`
- **Interpretation:** The correlation coefficient of approximately -0.017 indicates a very weak negative linear relationship between the product's rating and the review's helpfulness. This suggests that there is virtually no significant linear relationship between these two factors in the dataset.

### Most Helpful Reviewers

The analysis identified the top 10 reviewers with the most helpful votes. Here are their IDs along with the total number of helpful votes they received:

1. `A1MRPX3RM48T2I` - 2375 votes
2. `A5JLAU2ARJ0BO` - 2063 votes
3. `A2D1LPEUCTNT8X` - 2033 votes
4. `A3MQAQT8C6D1I7` - 1846 votes
5. `A15S4XW3CRISZ5` - 1470 votes
6. `A1N40I9TO33VDU` - 1142 votes
7. `A1UED9IWEXZAVO` - 1132 votes
8. `A250AXLRBVYKB4` - 1108 votes
9. `A680RUE1FDO8B` - 1101 votes
10. `A2IIN2NFYXHC4J` - 1092 votes

These reviewers are considered the most helpful based on the total number of helpful votes their reviews have received.

### Trend of Review Helpfulness Over Time

The trend of review helpfulness over time was plotted to observe how the average helpful votes changed year by year. The plot provides a visual representation of whether reviews have been getting more or less helpful over time.

<img width="630" alt="Screen Shot 2023-11-21 at 7 14 24 PM" src="https://github.com/joaovasco01/SentimentAnalysis_Ideas-Comments/assets/61276111/ce736650-7903-43bc-b5b9-a2763c14b4da">


Overall Decline: Following the initial peak, there is a general downward trend in the average number of helpful votes over time. This trend suggests that reviews are considered less helpful by users or that users are less inclined to vote on the helpfulness of reviews as time progresses.




# Review Helpfulness Prediction Model

## Model Summary

A binary classification model was built to predict whether an Amazon product review will be considered helpful. The helpfulness label was defined as binary, with reviews receiving more than 5 votes considered helpful (labeled as 1), and all others considered not helpful (labeled as 0).

## Code and Results

The model was trained using the following steps:

- Reviews with more than 5 helpful votes are labeled as 1 (helpful), else 0 (not helpful).
- The feature used for prediction is the review text.
- TF-IDF vectorization is applied to convert text to numeric features, limited to 3 features for simplicity.
- A RandomForestClassifier is used for training.
- The dataset is split into 80% training and 20% testing sets.

The classification report from the model is as follows:

              precision    recall  f1-score   support

           0       0.89      0.99      0.94     81337
           1       0.45      0.07      0.13     10551

    accuracy                           0.88     91888
   macro avg       0.67      0.53      0.53     91888
weighted avg       0.84      0.88      0.84     91888

## Evaluation

The model shows high precision for class 0 (not helpful), but low precision and recall for class 1 (helpful), indicating a model bias towards predicting not helpful reviews.

## Ideas for Improvement

- **Balancing the Dataset:** The dataset is highly imbalanced, with most reviews having 0 votes. This imbalance can lead to biased results, which is evident from the precision and recall scores. Techniques like SMOTE, undersampling, or assigning class weights in the model could be explored to address this imbalance.
- **Incorporating More Features:** Although the current model uses only 3 features due to computational constraints, including more features such as the length of the review, the sentiment score, and the time of the review could potentially improve the model's performance.
- **Preventing Overfitting:** Care must be taken not to overfit the model when adding more features. Cross-validation and regularization techniques should be used to ensure the model generalizes well to unseen data.
- **Additional Resources:** With more computational power and time, a more thorough grid search for hyperparameter tuning could be conducted, and more complex models or ensemble methods could be tested.

By addressing these points, we could improve the model's ability to accurately predict the helpfulness of reviews.


