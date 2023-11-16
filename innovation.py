import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Function to convert HTML to plain text
def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

# Function to calculate sentiment scores
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score['compound']  # Returning only the compound score

# Load Ideas JSON file
file_path_ideas = '/Users/joaovasco/Desktop/Part I/ideas.json'
df_ideas = pd.read_json(file_path_ideas)

# Apply HTML to text conversion and sentiment analysis on ideas
df_ideas['description'] = df_ideas['description'].apply(html_to_text)
df_ideas['sentiment_score'] = df_ideas['description'].apply(analyze_sentiment)

# Load Comments JSON file
file_path_comments = '/Users/joaovasco/Desktop/Part I/comments.json'
df_comments = pd.read_json(file_path_comments)

# Apply HTML to text conversion and sentiment analysis on comments
df_comments['body'] = df_comments['body'].apply(html_to_text)
df_comments['sentiment_score'] = df_comments['body'].apply(analyze_sentiment)

# Aggregate the comments' sentiment scores for each idea
comments_sentiment = df_comments.groupby('entityId')['sentiment_score'].mean().reset_index()
comments_sentiment.rename(columns={'entityId': 'id', 'sentiment_score': 'average_comment_sentiment'}, inplace=True)

# Merge the ideas DataFrame with the aggregated comments sentiment scores
df_merged = pd.merge(df_ideas, comments_sentiment, on='id', how='left')

# Display the DataFrame with idea and comments sentiment scores
print(df_merged[['id', 'description', 'sentiment_score', 'average_comment_sentiment']])




import matplotlib.pyplot as plt
import numpy as np

# ... [your existing code for loading and processing data] ...

# Set the width of the bars
bar_width = 0.35

# Set positions of bars on X axis
r1 = np.arange(len(df_merged['id']))
r2 = [x + bar_width for x in r1]

# Create the bar plot
plt.figure(figsize=(15, 8))  # Increased figure size for better visibility

plt.bar(r1, df_merged['sentiment_score'], color='blue', width=bar_width, edgecolor='grey', label='Idea Sentiment')
plt.bar(r2, df_merged['average_comment_sentiment'], color='yellow', width=bar_width, edgecolor='grey', label='Comment Average Sentiment')

# Add labels and title
plt.xlabel('Idea ID', fontweight='bold')
plt.ylabel('Sentiment Score', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(df_merged['id']))], df_merged['id'], rotation=45)  # Rotate labels
plt.ylim([-1, 1])
plt.title('Sentiment Scores of Ideas and Comments')

# Display only a subset of labels
plt.xticks([r + bar_width/2 for r in range(0, len(df_merged['id']), 5)])  # Adjust the step as needed

# Create legend & Show plot
plt.legend()
plt.show()
