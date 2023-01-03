import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

###########################
# Import data
###########################

reviews_df = pd.read_csv('amazon_reviews.csv')

###########################
# Exploratory data analysis
###########################

print(reviews_df)
print(reviews_df.info())
print(reviews_df.describe())

# Plot count plot for the ratings field 
# sns.countplot(x = reviews_df['rating'])
# plt.show()

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
print(reviews_df.describe())


# Plot historgram for length field
# reviews_df['length'].plot(bins=50, kind='hist', xlabel="Length of reviews")
# plt.show()

# Plot the count plot for the feedback field
# sns.countplot(x = reviews_df['feedback'])
# plt.show()

###########################
# Wordcloud
###########################

positive = reviews_df[reviews_df['feedback'] == 1]
negative = reviews_df[reviews_df['feedback'] == 0]

positive_sentences = positive['verified_reviews'].tolist()
print(len(positive_sentences))
positive_sentences_string  = ' '.join(positive_sentences)
# print(positive_sentences_string)
# plt.figure(figsize = (20, 20))
# plt.imshow(WordCloud().generate(positive_sentences_string))
# plt.show()

negative_sentences = negative['verified_reviews'].tolist()
print(len(negative_sentences))
negative_sentences_string = ' '.join(negative_sentences)
# print(negative_sentences_string)
# plt.figure(figsize = (20, 20))
# plt.imshow(WordCloud().generate(negative_sentences_string))
# plt.show()

###########################
# Data cleaning
###########################

# Define pipeline for data cleaning 
def message_cleaning(message):
    message_wo_punc = [char for char in message if char not in string.punctuation]
    message_wo_punc = ''.join(message_wo_punc)
    message_wo_punc_and_stopwords = [word for word in message_wo_punc.split() if word.lower() not in stopwords.words('english')]
    return message_wo_punc_and_stopwords

# Define vectorizer 
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

print(vectorizer.get_feature_names_out())
print(reviews_countvectorizer.shape)

reviews = pd.DataFrame(reviews_countvectorizer.toarray())
sentiment = reviews_df['feedback']

###########################
# Train and test model
###########################

reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(reviews, sentiment, test_size = 0.2)

# Naive Bayes classifier
nb_classifer = MultinomialNB()
nb_classifer.fit(reviews_train, sentiment_train)

sentiment_predict_test = nb_classifer.predict(reviews_test)

# Plot confusion matrix
# cm = confusion_matrix(sentiment_test, sentiment_predict_test)
# sns.heatmap(cm, annot=True)
# plt.show()

# Print classification report
print(classification_report(sentiment_test, sentiment_predict_test))

# Logistic Regression
model = LogisticRegression()
model.fit(reviews_train, sentiment_train)

sentiment_predict_test = model.predict(reviews_test)

# Plot confusion matrix
# cm = confusion_matrix(sentiment_test, sentiment_predict_test)
# sns.heatmap(cm, annot=True)
# plt.show()

# Print classification report
print(classification_report(sentiment_test, sentiment_predict_test))


# Gradient Boost 
model = GradientBoostingClassifier()
model.fit(reviews_train, sentiment_train)

sentiment_predict_test = model.predict(reviews_test)

# Plot confusion matrix
# cm = confusion_matrix(sentiment_test, sentiment_predict_test)
# sns.heatmap(cm, annot=True)
# plt.show()

# Print classification report
print(classification_report(sentiment_test, sentiment_predict_test))
