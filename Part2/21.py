import pandas as pd

# Load the dataset
data = pd.read_json('/Users/joaovasco/Desktop/Part I/Part2/Software.json', lines=True)

# Convert 'vote' to numeric, handling non-numeric values
data['vote'] = pd.to_numeric(data['vote'], errors='coerce')

# Calculate the correlation
correlation = data['overall'].corr(data['vote'])
print("Correlation:", correlation) # indicates a very weak negative linear relationship 

#1. Correlation Between Product Rating and Review Helpfulness
#This analysis aims to determine if there is a statistical relationship between the rating a product receives (overall field) and how helpful users find the review (vote field). The corr function in Pandas calculates this correlation. A positive correlation would mean that higher-rated products tend to have more helpful reviews, whereas a negative correlation would indicate the opposite. No correlation would suggest that the product's rating does not significantly impact the perceived helpfulness of its reviews.

# Group by reviewerID and sum their votes (considered most helpful as the ones with the ost amount of votes)
helpful_reviewers = data.groupby('reviewerID')['vote'].sum().sort_values(ascending=False)

# Display top 10 most helpful reviewers
print(helpful_reviewers.head(10))

import matplotlib.pyplot as plt

# Convert reviewTime to datetime
data['reviewTime'] = pd.to_datetime(data['reviewTime'])

# Group by year and calculate average votes
average_votes_per_year = data.groupby(data['reviewTime'].dt.year)['vote'].mean()

print("Correlation:", correlation)


# Plot 
plt.plot(average_votes_per_year)
plt.xlabel('Year')
plt.ylabel('Average Helpful Votes')
plt.title('Trend of Review Helpfulness Over Time')
plt.show()
