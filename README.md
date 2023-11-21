## Sentiment Analysis of Ideas and Comments

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


