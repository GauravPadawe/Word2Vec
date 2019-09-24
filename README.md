# Word2Vec using Gensim

### Source :

Download : https://www.kaggle.com/c/word2vec-nlp-tutorial/data

### Table of Contents

#### 1. **Information**
    - Details
    - Objective

#### 2. **Loading Dataset**
    - Importing packages
    - Reading Data
    - Shape of data
    - Dtype

#### 3. **Text Cleansing**

#### 4. **Modeling**

#### 5. **Saving and Loading**

#### 6. **Conclusion**

#### 7. **What's next ?**

#### 8. **References**<br>

### Details :

- The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

- labeledTrainData - The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  


- testData - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one. 


- unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 


- sampleSubmission - A comma-delimited sample submission file in the correct format.


- **Data fields :**
    - id - Unique ID of each review
    - sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
    - review - Text of the review


### Objective :

- The goal is to build Word2Vec Word Embedding model with the help of Gensim for NLP Tasks.
