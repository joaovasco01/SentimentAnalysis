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

### Conclusion
This script effectively demonstrates how sentiment analysis can be applied to real-world data, providing a straightforward and visual approach to understanding complex emotional data.
