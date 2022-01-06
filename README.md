# Sentimental-Analysis-using-Amazon-Review-Data

Predicting a review's numeric rating based on the textual review is a quintessential multiclass text classification problem and an interesting research topic in ```Natural Language Processing```. New advances in NLP including the development of Glove, and Word2Vec, have increased the range of approaches available to address this question, and insights gained on this problem can generalize across sentiment analysis and NLP multiclass classification problems in general. We leveraged the extensive corpus of ```multi-class labeled Amazon data to apply sentiment analysis```.

### Objective:
     Given a text book review, predict one of the three (positive, neutral, negative) sentiment classes.

This project was completed using Jupyter Notebook and Python with Pandas, NumPy, Matplotlib, Scikit-Learn, nltk, spacy and textblob.

`Model Deployment Link:`
  https://share.streamlit.io/samarth3557/sentimental-analysis-using-amazon-review-data/main/streamlit.py


### Features
- unique_id
- asin
- product_name
- product_type
- helpful	rating
- title
- date
- reviewer
- reviewer_location
- review_text

## Exploratory Data Analysis

First, we processed the data of review column with following steps:

    1. Lower Casing and Removed punctuations by defining a new function.
    
    2.With nltk, we imported stop words and removed them from data cleaned so far, like-i,me,myself,himself,she etc.
    
    3.We removed frequent words from data by defining a new function.
    
    4.Then we stemmarised the data, which means to presenting the summary of generated data in an easily comprehensible
       and informative manner.
       
    5.We lemmatizd the data, which is the process of grouping together the inflected forms of a word so they can be analysed as
      a single item, identified by the words lemma
      
    6.Then we removed urls and tags from the data.
    
    7.After all above data processing steps we calculated the polarity and subjective of each review with analysis labels 
      (positive-more than 0, negative-less than 0 and neutral-equal to 0)
      
        Positive    30296
        Negative     8659
        Neutral      3045
        
    8.We started data modeling, divided the data 33600 training and 8400 testing 
    
    9.After vectorising, we got Accuracy score on training dataset= 0.9995535714285714 and Accuracy on 
      Test dataset=0.9942857142857143
      
    10.We performed prediction on test data and got F1 score= 0.9871397964692025
    
    
![Image of result](https://github.com/Technocolabs100/Multi-Class-Text-Sentiment-Analysis-Using-Amazon-Review-Data/blob/main/Model%20Deployment%20-%20Streamlit/Images/Demo.png)
    
  ### Results:
  ![image of result](https://github.com/Technocolabs100/Multi-Class-Text-Sentiment-Analysis-Using-Amazon-Review-Data/blob/main/Model%20Deployment%20-%20Streamlit/Images/Analysis.png)

 ### Conclusion:       
   This complete notebook represents an exploratory data analysis (EDA) process to put together the patterns related to the Reviews. We have dealt with the reviews data `"multi-class labeled Amazon data to apply sentiment analysis"` and performed sentimental analysis using the textblob to calculate the (positive, neutral, negative) polarity values.  The purpose of this study is to analyze the dataset and obtain important insights form it. After performing the Data preprocessing tasks, we have developed an model using the `KNN Algorithm` for calculating the accuracy of the model. Further, using the `streamlit web application` we have deployed our model to predict our results.

### Team
- **Samarth**
- **Arjun**

### Mentor
- **Uttej Kumar**
