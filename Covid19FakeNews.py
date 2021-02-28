#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv(r'C:\Users\mhdha\Desktop\corona_fake.csv')


# In[2]:


df.shape


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.fillna('', inplace=True)
df.head()


# In[6]:


df['label'].value_counts()


# ## Label pre-processing

# In[7]:


df.loc[df['label'] == 'fake', 'label'] = 'FAKE'
df.loc[df['label'] == 'Fake', 'label'] = 'FAKE'
df['label'].value_counts()


# In[8]:


# Drop empty string
df = df[df['label']!='']
df.shape


# In[9]:


def convert_label(label):
    if label =='FAKE':
        return 0
    elif label =='TRUE':
        return 1
    
df['label'] = df['label'].apply(convert_label)
df['label'].value_counts()


# ## Text pre-processing

# In[10]:


def clean(text):
    # Set of stopwords in English
    stop_words = set(stopwords.words('english'))
    
    # delete numbers
    text = re.sub('[^a-zA-Z]',' ', text)
    
    # All in lower case
    text = text.lower()
    
    # delete html tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # delete twitter usernames
    text = re.sub(r'@[A-Za-z0-9]+','', text)
    
    # delete urls
    text = re.sub('https?://[A-za-z0-9]','', text)
    
    # delete numbers
    text = re.sub('[^a-zA-Z]',' ', text)
    
    # seperate strings
    word_tokens = word_tokenize(text)
    
    # delete stopwords
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
            
    text = (' '.join(filtered_sentence))
    return text


# In[11]:


word_tokenize('Hello world 1 2')


# In[12]:


for column in df.columns:
    if column != 'label':
        df[column] = df[column].apply(clean)        
df.head()


# In[13]:


# Train_Test split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:2], df['label'], test_size=0.2, random_state=11)

print(X_train.shape)
print(X_test.shape)


# In[14]:


# Vectorize
title_vectorizer = CountVectorizer()
text_vectorizer = CountVectorizer()


# In[15]:


X_train_title = title_vectorizer.fit_transform(X_train['title']).toarray()
X_train_text = text_vectorizer.fit_transform(X_train['text']).toarray()

print('Title shape\t:', X_train_title.shape)
print('Title text\t:', X_train_text.shape)


# In[16]:


X_test_title = title_vectorizer.transform(X_test['title']).toarray()
X_test_text = text_vectorizer.transform(X_test['text']).toarray()

print(X_test_title.shape)
print(X_test_text.shape)


# In[17]:


X_train_title_text = np.hstack((X_train_title, X_train_text))
X_test_title_text = np.hstack((X_test_title, X_test_text))

print(X_train_title_text.shape)
print(X_test_title_text.shape)


# ## Machine Learning model

# In[18]:


clf = MultinomialNB()
clf.fit(X_train_title_text, y_train)
print('Accuracy on train data\t:', clf.score(X_train_title_text, y_train))
print('Accuracy on test data\t:', clf.score(X_test_title_text, y_test))


# ## Testing

# In[21]:


test_title = 'Covid 19 found in toilet paper'
test_text = 'strain of deadly virus breeds rapidly in tissue-fibres'


# In[20]:


test_title = clean(test_title)
test_text = clean(test_text)

print(test_title)
print(test_text)


# In[25]:


test_title_vec = title_vectorizer.transform([test_title]).toarray()
test_text_vec = text_vectorizer.transform([test_text]).toarray()

print(X_test_title_vec.shape)
print(X_test_text_vec.shape)


# In[27]:


test_title_text = np.hstack((test_title_vec, test_text_vec))
print(test_title_text.shape)


# In[28]:


print('Prediction \t\t:', clf.predict(test_title_text))
print('Prediction per class \t\t:', clf.predict_proba(test_title_text))


# The output is 0, which means the news is fake with probability about 73.35%
