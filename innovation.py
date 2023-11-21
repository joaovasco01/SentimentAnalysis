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














import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



# Text Preprocessing
def preprocess_text(text):
    return text.lower()

# Apply preprocessing to the descriptions
df_ideas['processed_description'] = df_ideas['description'].apply(preprocess_text)

# Vectorize the text descriptions
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_ideas['processed_description'])

# Cluster the ideas based on their vectorized descriptions
kmeans = KMeans(n_clusters=6, random_state=42)
df_ideas['cluster'] = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(12, 8))
palette = sns.color_palette("hsv", n_colors=6)
sns.scatterplot(data=df_ideas, x='id', y='cluster', hue='cluster', palette=palette, s=100, legend='full')
plt.title('Clustering of Ideas Based on Descriptions')
plt.xlabel('Idea ID')
plt.ylabel('Cluster')
plt.legend(title='Cluster')
plt.show()

# Identify the most common cluster
most_common_cluster = df_ideas['cluster'].value_counts().idxmax()

# Filter the DataFrame to get IDs and descriptions of ideas in the most common cluster
most_common_cluster_data = df_ideas[df_ideas['cluster'] == most_common_cluster][['id', 'description']]

# Write the IDs and descriptions to a file
with open('cluster_descriptions.txt', 'w') as file:
    id_list = []
    for index, row in most_common_cluster_data.iterrows():
        id_list.append(row['id'])
        file.write(f"ID: {row['id']}\nDescription: {row['description']}\n\n")

print("IDs and descriptions of ideas in the most common cluster have been written to 'cluster_descriptions.txt'")
print(id_list)


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import seaborn as sns

# # Text Preprocessing
# def preprocess_text(text):
#     return text.lower()

# # Apply preprocessing to the descriptions
# df_ideas['processed_description'] = df_ideas['description'].apply(preprocess_text)

# # Vectorize the text descriptions
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(df_ideas['processed_description'])

# # Cluster the ideas based on their vectorized descriptions
# kmeans = KMeans(n_clusters=5, random_state=42)
# df_ideas['cluster'] = kmeans.fit_predict(X)

# # Merge the ideas DataFrame with the aggregated comments sentiment scores
# df_merged = pd.merge(df_ideas, comments_sentiment, on='id', how='left')

# # Visualization: Scatter Plot for Clusters
# plt.figure(figsize=(12, 8))
# palette = sns.color_palette("hsv", n_colors=len(df_ideas['cluster'].unique()))
# sns.scatterplot(data=df_merged, x='id', y='cluster', hue='cluster', palette=palette, s=100, legend='full')
# plt.title('Clustering of Ideas Based on Descriptions')
# plt.xlabel('Idea ID')
# plt.ylabel('Cluster')
# plt.legend(title='Cluster')
# plt.show()





# # Identify the most common cluster
# most_common_cluster = df_merged['cluster'].index[0]

# # Filter the DataFrame to get IDs and descriptions of ideas in the most common cluster
# most_common_cluster_data = df_merged[df_merged['cluster'] == most_common_cluster][['id', 'description']]

# # Write the IDs and descriptions to a file
# with open('cluster_descriptions.txt', 'w') as file:
#     for index, row in most_common_cluster_data.iterrows():
#         list.append(row['id'])
#         file.write(f"ID: {row['id']}\nDescription: {row['description']}\n\n")

# print("IDs and descriptions of ideas in the most common cluster have been written to 'cluster_descriptions.txt'")
# print(list)
# [324, 361, 515, 642, 769]
# service [100, 101, 144, 152, 155, 157, 185, 193, 194, 196, 197, 230, 261, 262, 263, 267, 295, 299, 301, 311, 358, 359, 385, 386, 417, 418, 420, 423, 425, 427, 449, 450, 453, 455, 456, 457, 459, 482, 483, 481, 484, 485, 487, 546, 547, 549, 609, 673, 706, 961, 962, 1025, 1057, 1121]