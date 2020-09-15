#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import nltk 
import string
import re


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import random

import pickle


# In[3]:


get_ipython().system('gdown --id 1sPDkfIWZPmCd_IsHjO9SXIFJreOI0S21')


# In[11]:


df = pd.read_csv('app_reviews.csv')
len(df)


# In[12]:


len(df)


# In[13]:


positive_features=df.loc[(df['sentiment']=='Positive'),['content']]
negative_features=df.loc[(df['sentiment']=='Negative'),['content']]
neutral_features=df.loc[(df['sentiment']=='Neutral'),['content']]


# In[14]:


def tokenization(text):
    text = re.split('\W+', text)
    return text


# In[15]:


#apply tokenization
positive_features_processed= positive_features['content'].apply(lambda x: tokenization(x)).tolist()
negative_features_processed= negative_features['content'].apply(lambda x: tokenization(x)).tolist()
neutral_features_processed= neutral_features['content'].apply(lambda x: tokenization(x)).tolist()


# In[16]:


len(negative_features_processed)


# In[17]:


def create_word_features(words):
    
    # Remove all stopwords
    useful_words = [word for word in words if word not in stopwords.words("english")]
    
    # For each word, we create a dictionary with all the words and True. 
    # Why a dictionary? So that words are not repeated. If a word already exists, it won’t be added to the dictionary.
    my_dict = dict([(word, True) for word in useful_words])
    
    return my_dict


# In[18]:


positive_reviews = []
for i in range(0,3250):
    words = positive_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    positive_reviews.append((create_word_features(words), "Positive"))


# In[19]:


print( positive_reviews[985])


# In[20]:


negative_reviews = []
for i in range(0,2962):
    # We get all the words in that file.
    words = negative_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    negative_reviews.append((create_word_features(words), "Negative"))


# In[21]:


print( negative_reviews[1064])


# In[22]:


neutral_reviews = []
for i in range(0,2888):
    # We get all the words in that file.
    words = neutral_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    neutral_reviews.append((create_word_features(words), "Neutral"))


# In[23]:


print(neutral_reviews[280])


# In[24]:




reviews = positive_reviews + negative_reviews + neutral_reviews
random.shuffle(reviews)

train_data = reviews[:6371]
test_data = reviews[6371:]


# In[25]:


## Let’s create our Naive Bayes Classifier, and train it with our training set.
classifier = NaiveBayesClassifier.train(train_data)

# And let’s use our test set to find the accuracy
accuracy = nltk.classify.util.accuracy(classifier, test_data)
print("Accuracy is:",accuracy*100 )
print(classifier.show_most_informative_features(10))


# In[26]:


#save the model using pickle
save_classifier=open('model.pickle','wb')
pickle.dump(classifier,save_classifier)
save_classifier.close()


# In[27]:


#load the the model
load_classifier=open('model.pickle','rb')
classifier=pickle.load(load_classifier)
load_classifier.close()


# In[28]:


#test
custom_review = "très bonne application"
words = word_tokenize(custom_review)
words = create_word_features(words)
classifier.classify(words)

