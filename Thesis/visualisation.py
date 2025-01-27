import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

class Visualisation:
    def __init__(self, categories_order=None):
        if categories_order is None:
            self.categories_order = [
                "unknown", 
                "strongly negative", 
                "negative", 
                "weakly negative", 
                "neutral", 
                "weakly positive", 
                "positive", 
                "highly positive"
            ]
        else:
            self.categories_order = categories_order

    def convert_to_sentiment_category(self, score):
        score = float(score)
        if 2.0 <= score <= 3.0:
            return 'highly positive'
        elif 1.0 <= score < 2.0:
            return 'positive'
        elif 0.0 < score < 1.0:
            return 'weakly positive'
        elif score == 0.0:
            return 'neutral'
        elif -1.0 < score <= 0.0:
            return 'weakly negative'
        elif -2.0 < score <= -1.0:
            return 'negative'
        elif -3.0 <= score <= -2.0:
            return 'strongly negative'
        else:
            return 'unknown'

    def plot_sentiment_histogram(self, csv_file):
        data = pd.read_csv(csv_file)

        # Extract the labels (assumes the label is the last column)
        labels = data.iloc[:, -1].tolist()

        labels = [re.sub(r"\[(.*?)\]", r'\1', label) for label in labels]
        labels = [re.sub(r"\'(.*?)\'", r'\1', label) for label in labels]
        labels = [float(label) if isinstance(label, str) else np.nan for label in labels]

        # Save the labels to a CSV file
        labels_df = pd.DataFrame(labels, columns=['Labels'])
        labels_df.to_csv('check.csv', index=False)
        print("Labels saved to 'check.csv'")

        sentiment_categories = [self.convert_to_sentiment_category(score) for score in labels]

        # Count the occurrences of each sentiment category
        sentiment_count = Counter(sentiment_categories)

        # Ensure all categories are present in the count, even if some have 0 count
        sentiment_count = {category: sentiment_count.get(category, 0) for category in self.categories_order}

        # Prepare data for plotting
        categories = list(sentiment_count.keys())
        counts = list(sentiment_count.values())

        # Create the histogram plot
        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts, color='skyblue')

        # Customize the plot
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.title('Sentiment Distribution')
        plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        plt.tight_layout()  # Adjust layout to avoid clipping

        # Save the plot as 'histogram.png' in the 'plots' directory
        plt.tight_layout()
        plt.savefig('plots/histogram.png')
