#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().system('gdown --id 1u5O9VjGvYJosxdEZ5Ue0D5o9HEe9hJqa')


# In[3]:


df = pd.read_csv('app_reviews.csv')
len(df)


# In[4]:


from google_play_scraper import Sort, reviews, app ##for scrapping app contents from google store
import pandas as pd
import numpy as np
from tqdm import tqdm

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.figure_factory as ff

#plotly.offline doesn't push your charts to the clouds
import plotly.offline as pyo
pyo.offline.init_notebook_mode()


# In[5]:


# lets see the sentiments value by app
sentiments = df.groupby(['title','sentiment']).sentiment.count().unstack()
sentiments.plot(kind='bar',title='Sentiments by App')


# In[6]:


len(df)


# In[7]:


positive_features=df.loc[(df['sentiment']=='Positive'),['content']]
negative_features=df.loc[(df['sentiment']=='Negative'),['content']]
neutral_features=df.loc[(df['sentiment']=='Neutral'),['content']]


# In[8]:


def tokenization(text):
    text = re.split('\W+', text)
    return text


# In[9]:


#apply tokenization
positive_features_processed= positive_features['content'].apply(lambda x: tokenization(x)).tolist()
negative_features_processed= negative_features['content'].apply(lambda x: tokenization(x)).tolist()
neutral_features_processed= neutral_features['content'].apply(lambda x: tokenization(x)).tolist()


# In[18]:


len(neutral_features_processed)


# In[11]:


def create_word_features(words):
    
    # Remove all stopwords
    useful_words = [word for word in words if word not in stopwords.words("english")]
    
    # For each word, we create a dictionary with all the words and True. 
    # Why a dictionary? So that words are not repeated. If a word already exists, it won’t be added to the dictionary.
    my_dict = dict([(word, True) for word in useful_words])
    
    return my_dict


# In[14]:


positive_reviews = []
for i in range(0,4000):
    words = positive_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    positive_reviews.append((create_word_features(words), "Positive"))


# In[15]:


print( positive_reviews[985])


# In[17]:


negative_reviews = []
for i in range(0,3752):
    # We get all the words in that file.
    words = negative_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    negative_reviews.append((create_word_features(words), "Negative"))


# In[21]:


print( negative_reviews[1064])


# In[19]:


neutral_reviews = []
for i in range(0,3543):
    # We get all the words in that file.
    words = neutral_features_processed[i]
    # Then we use the function we wrote earlier to create word features in the format nltk expects.
    neutral_reviews.append((create_word_features(words), "Neutral"))


# In[23]:


print(neutral_reviews[280])


# In[26]:




reviews = positive_reviews + negative_reviews + neutral_reviews
random.shuffle(reviews)

train_data = reviews[:8000]
test_data = reviews[8000:]


# In[27]:


## Let’s create our Naive Bayes Classifier, and train it with our training set.
classifier = NaiveBayesClassifier.train(train_data)

# And let’s use our test set to find the accuracy
accuracy = nltk.classify.util.accuracy(classifier, test_data)
print("Accuracy is:",accuracy*100 )
print(classifier.show_most_informative_features(10))


# In[28]:


#save the model using pickle
save_classifier=open('model.pickle','wb')
pickle.dump(classifier,save_classifier)
save_classifier.close()


# In[29]:


#load the the model
load_classifier=open('model.pickle','rb')
classifier=pickle.load(load_classifier)
load_classifier.close()


# In[25]:


#test
custom_review = "pas mal"
words = word_tokenize(custom_review)
words = create_word_features(words)
classifier.classify(words)


# In[ ]:




