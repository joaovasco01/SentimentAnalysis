import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline

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

# Initialize the zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["product", "service", "business model", "work practice change", "marketing", "cost saving"]

# Function to classify idea descriptions
def classify_idea(text):
    result = classifier(text, candidate_labels)
    return result['labels'][0]  # Returning the top label

# Apply zero-shot classification to ideas
df_ideas['innovation_type'] = df_ideas['description'].apply(classify_idea)

# Output file path
output_file_path = '/Users/joaovasco/Desktop/Part I/Bert_results.txt'

# Save the DataFrame with idea, sentiment scores, and innovation type to a CSV file
df_ideas[['id', 'innovation_type']].to_csv(output_file_path, index=False)
