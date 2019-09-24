#!/usr/bin/env python
# coding: utf-8

# # Word2Vector using Gensim
# 
# ### Table of Contents
# 
# #### 1. **Information**
#     - Details
#     - Objective
# 
# #### 2. **Loading Dataset**
#     - Importing packages
#     - Reading Data
#     - Shape of data
#     - Dtype
# 
# #### 3. **Text Cleansing**
# 
# #### 4. **Modeling**
# 
# #### 5. **Saving and Loading**
# 
# #### 6. **Conclusion**
# 
# #### 7. **What's next ?**
# 
# #### 8. **References**<br>
# 
# ### Source :
# - https://www.kaggle.com/c/word2vec-nlp-tutorial/data
# 
# ### Details :
# 
# - The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.
# 
# - labeledTrainData - The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  
# 
# 
# - testData - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one. 
# 
# 
# - unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 
# 
# 
# - sampleSubmission - A comma-delimited sample submission file in the correct format.
# 
# 
# - **Data fields :**
#     - id - Unique ID of each review
#     - sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
#     - review - Text of the review
# 
# 
# ### Objective :
# 
# - The goal is to build Word2Vec Word Embedding model with the help of Gensim for NLP Tasks.

# ### Load dataset

# In[ ]:


#Import packages

import gensim
#import warnings
import re, string
import pandas as pd
#warnings.filterwarnings(action='ignore')


# - We are loading Gensim here which is fairly straightforward library / module to build Word2Vec models.
# 
# 
# - We'll see it in more details as we move along, In case the gensim isn't installed use this command **"!pip install gensim"**

# In[43]:


#reading dataset

df = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
df.head()


# In[44]:


#shape of dataset
print ('Shape od Dataset :' ,df.shape)


# In[5]:


#Lets review 1st corpus
df['review'][0]


# - We can observe that our corpus may contain HTML tags, special characters, etc.
# 
# 
# - We need to clean the corpus by eliminating these tags and characters.

# In[6]:


#checking dtypes
df.dtypes


# ### Data Cleaning

# In[ ]:


#importing beautiful soup
from bs4 import BeautifulSoup

#empty list
clean = [ ]

#for loop to clean dataset while removing html tags, special characters
for doc in df['review']:
    x = doc.lower()                     #lowerthe case
    x = BeautifulSoup(x, 'lxml').text   #html tag removal
    x = re.sub('[^A-Za-z0-9]+', ' ', x) #replacing it by space to seperate words
    clean.append(x)

#assigning clean list to new attribute
df['clean'] = clean


# - Above we're using beautiful soup library to eliminate HTML tags. Alternate way to this is we can use Regex (Regular Expression) to eliminate tags.
# 
# 
# - Finally, we are appending the clean set to list and forming a new attribute as clean.

# In[9]:


#viewing data
df.head()


# - Above we can observe that we had succesfully cleaned the data set.
# 
# 
# - Now, its ready to pass into Word2Vec but lets learn about Word2Vec Word Embeddings :
# 
# 
#     - Where and Why ?

# In[10]:


#tokenizing the corpus

docs = [ ]

for doc in df['clean']:
    docs.append(doc.split(' '))

print ('First Doc :', docs[0])


# - Above we had Tokenized the data where we're actually splitting the words on individual level.

# ### Word Embeddings
# 
# 
# - Word2Vec is a model to form / create word Embeddings. Its a modern way of representing Word , wherein each word is represented by a vector (Array of numbers based on Embedding Size). Vectors are nothing but the weights of neurons, so if we set neurons of size 100 then we will have 100 weights and those weights are our Word Embeddings or simply Dense vector.
# 
# 
# - Input word must be a One hot encoded. For example :
# 
# 
# ![alt_text](https://miro.medium.com/max/714/1*UOjWvDziH86T2MmiDpp98Q.png)
# 
# 
# - **Why Word2Vec ? :** Word2Vec finds relation (Semantic or Syntactic) between the words which was not possible by our Tradional TF-IDF or Frequency based approach. When we train the model, each one hot encoded word gets a point in a dimensional space where it learns and groups the words with similar meaning. 
# 
# 
# - The neural network incorporated here is a Shallow.
# 
# 
# - One thing to note here is that we need large textual data to pass into Word2Vec model in order to figure out relation within words or generate meaningful results.
# 
# 
# - In general the Word2Vec is based on Window Method, where we have to assign a Window size. 
# 
# 
# ![alt_text](https://1.bp.blogspot.com/-nZFc7P6o3Yc/XQo2cYPM_ZI/AAAAAAAABxM/XBqYSa06oyQ_sxQzPcgnUxb5msRwDrJrQCLcBGAs/s1600/image001.png)
# 
# 
# - In above visual representation , Window size is set to 1. So, 1 word from both the sides of target are being considered. Similarly, in each iteration, window will slides by single stride and our neighbors will keep changing. 
# 
# 
# - There are 2 types of Algorithms : CBOW and Skip-gram.
# 
# #### (a) Continuous Bag of Words (CBOW) :
# 
# - In here neighboring words are provided as Input to predict the Target. In other words, A context is Provided as input to predict the Target. For example :
# 
# 
# - Let us look at this in visual representation :
# 
# 
# ![alt_text](https://www.researchgate.net/publication/283531484/figure/fig2/AS:667840583577604@1536237011465/Continuous-Bag-of-Words-model.png)
# 
# 
# - We can observe that **W(t)** is being predicted given that the context / neighboring words are feeded as input. P-value is estimated for target word based on the input that is feeded to the network.
# 
# 
# - We've a shallow neural network here. Size of each word embedding or individual dense vector is depended on the number of neurons we've at disposal.
# 
# 
# - These vectors are weights learnt by each neuron. 
# 
# 
# - Regardless of how big or small the word is, it will be represented by size of embedding we set.
# 
# 
# - **Advantages :**
# 
#     - CBOW is faster and represents frequent words more better.
#     - When it comes to memory utilization, CBOW tends to consume Low memory.
#     
#     
# - **Disadvantages :**
#     
#     - Fails to Represent Infrequent words.
#     - Needs huge textual data. (***Note : Can't say it is disadvantage since it consumes low memory but thought its worth mentioning***)
# 
# 
# #### (b) Skip-gram :
# 
# 
# - Skip-gram is opposite / inverse of CBOW, wherein a target word is provided as output in order to predict the Contextual / Neighboring words.
# 
# 
# - Let us look at this in visual representation :
# 
# 
# ![alt_text](https://sakhawathsumit.github.io/sumit.log/assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/skip-gram.png)
# 
# 
# - Here, a Target word is feeded to our shallow neural network, weights are learnt from hidden layer. One word is picked at random from the neighboring words (which are based on our Window Size). Furthermore, P-value is being calculated for context words being close to the input word we feed to the network.
# 
# 
# - **Advantages :**
# 
#     - Skip-gram works well with smaller datasets.
#     - Its able to represent rare words well.
#     
#     
# - **Disadvantages :**
#     
#     - Slow on training if dataset is huge.
#     - And is not memory efficient.
# 
# 
# ### But there is a PROBLEM !!!!
# 
# - Both techniques are based on probability estimation.
# 
# 
# - Let us consider a example here, we've a corpus of 1000 words. We need to predict either a target word (CBOW) or context (Skip-gram). Now, we specify a fixed Window size for sliding through the corpus. Let's say our Window size is 1 , i.e, In case of CBOW the Input will be 2 Words (Words on both sides of target) while in case of Skip-gram input will be a single word in order to predict context (2 words).
# 
# 
# - In this case the probabilities will be estimated for words within a certain window and those probabilites will be higher than that for those words which are far from neighborhood. 
# 
# 
# - Computing probabilities for huge corpus (e.g.: Millions of words) would be computationally expensive since algorithm will estimate probabilities for all the words at each iteration. In this case, The model may give irrelevant results too. And so to overcome this issue **Negative Sampling** was proposed.
# 
# 
# - In **Negative Sampling** the probabilities are estimated for word within our fixed Window size, just like before but only few random words are chosen out of fixed window. By this the load on p-value estimation is low compared to what it was before. Training is faster and results are more satisfactory.
# 
#     
# #### CBOW or Skip-gram - which to use when ?
# 
# - It depends on nature of problem.
# 
# 
# - Both have their own set of benefits.
# 
# 
# #### Gensim :
# 
# - Gensim is fairly easy to use module which inherits CBOW and Skip-gram.
# 
# 
# - We can install it by using **!pip install gensim**.
# 
# 
# - Alternate way to implement Word2Vec is to build it from scratch which is quite complex.
# 
# 
# - Read more about Gensim : https://radimrehurek.com/gensim/index.html
# 
# 
# - **FYI, Gensim was developed and is maintained by the NLP researcher Radim Řehůřek and his company RaRe Technologies.**

# ### Word2Vec Modeling
# 
# 
# - Further we'll look how to implement Word2Vec to oull out Word Embeddings.

# In[ ]:


#Word2vec implementation
model = gensim.models.Word2Vec(docs, # document (list of list)
                              min_count=10, #Ignore those words with total frequency less than 10
                              workers=4, #Number of CPU
                              size=50, #embedding size
                              window=5, #Maximum distance between current and predicted word
                              iter = 10)


# Here are few parameters which one could play with :
# 
# - **sentences :** The sentences accepts a list of lists of tokens. (Better to have large document list in case of CBOW)
# 
# 
# - **size :** Number of Neurons to incorporate in hidden layer or size of Word Embeddings. By default its set to 100.
# 
# 
# - **window :** Window Size or Number of words to consider around target. If size = 1 then 1 word from both sides will be considered. By default 5 is fixed Window Size.
# 
# 
# - **min_count :** Default value is 5. Words which are infrequent based on minimum count mentioned will be ignored.
# 
# 
# - **workers :** Number of CPU Threads to use at once for faster training.
# 
# 
# - **sg :** Either 0 or 1. Default is 0 or CBOW. One must explicitly define Skip-gram by passing 1.
# 
# 
# - **iter :** Number of Iterations or Epochs.

# In[20]:


#vocab size
len(model.wv.vocab.keys())


# - We've 28656 words in our vocabulary for usage after eliminating words which appeared less than 10.
# 
# 
# - And each word will be represented by 50 numbers / weights (which is our Embedding size).

# In[1]:


#model.wv.syn0.shape


# In[ ]:


#uncomment to view vocabulary
#model.wv.vocab


# In[23]:


#Lets look at vector of great
model.wv['great']


# - Above we can observe that "great" word is represented by 50 Weights / Embeddings.

# In[24]:


#find similar words to the given word
model.wv.most_similar('dumb')


# - One beautiful thing about Gensim is that we have such methods which can give us similar words aligned by highest probability.
# 
# 
# - Cosine Similarity is computed to between the words to find most similar words.
# 
# 
# - As we passed "dumb" we can observe that how most similar words are aligned by p-value.
# 
# 
# - Given "dumb" , most similar words are stupid, lame, silly, etc.

# In[26]:


#words which doesn't match
model.wv.doesnt_match('house rent trust apartment'.split())


# - Which document from the given list doesn’t go with the others from the training set ?
# 
# 
# - We passed "house", "rent", "trust" and "apartment" and result is "trust" , being not similar to other words and its satisfactory.

# In[41]:


#arithmetic operations
model.most_similar(positive=['woman','man'], negative=['king'])[0]


# - We can also perform basic arithmetic problems here.
# 
# 
# - If Woman + Man = King - ?
# 
# 
# - Output is "Girl" having p-value 0.74 , which is acceptable, If we would have larger corpora then we may get output as "Queen".

# In[38]:


#word embeddings
model.wv.vectors


# - Above is our array having 50 dimensions.
# 
# 
# - Which we can further use for Sentiment Analysis.

# ### Saving and Loading Model
# 
# 
# - We can save our model as a bin and review the content.
# 
# 
# - Further, we can also load the model using load() method.

# In[ ]:


#saving model
model.wv.save('word2vector-50.bin')


# In[ ]:


#loading model
model = gensim.models.Word2Vec.load('word2vec-50.bin')


# - We can also load Google Word2Vec model by downloading file from here : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# 
# 
# - It is a pre-trained model by Google which was trained on Google News Data , where each word is represented by 300 embedding size.
# 
# 
# - There are many such pre-trained models available, few to name are GloVe (Global Vectors for Word Representation) by Stanford, FastText by Facebook AI Research Lab. One can load them and examine which works better.

# ### Conclusion
# 
# 
# - In this case study we learnt CBOW and Skip-gram in detail.
# 
#  
# - We also seen what obstacle we were facing and how researchers overcame it by adding Negative Sampling.
# 
# 
# - We trained the model by passing a clean tokenized corpora.
# 
# 
# - Finally, we looked at how we can Save / Load model and use it in future.

# ### What's next ?
# 
# 
# - We can use word embeddings as feature for Sentiment Analysis. How ? We'll see that in next Case study.

# ### References
# 
# 
# - Read about FastText here : https://fasttext.cc/
# 
# 
# - Read about GloVe here : https://nlp.stanford.edu/projects/glove/
# 
# 
# - Read about GoogleNews Word2Vec here : https://code.google.com/archive/p/word2vec/
# 
# 
# - Wikipedia : https://en.wikipedia.org/wiki/Word2vec
# 
# 
# - **Mentions :** Images used here are imported from various other sources like ResearchGate, MiroMedium, RaRe Technologies and Others.
