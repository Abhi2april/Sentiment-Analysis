# Sentimental-Analysis-using-Amazon-Review-Data

Predicting a review's numeric rating based on the textual review is a quintessential multiclass text classification problem and an interesting research topic in ```Natural Language Processing```. New advances in NLP including the development of Glove, and Word2Vec, have increased the range of approaches available to address this question, and insights gained on this problem can generalize across sentiment analysis and NLP multiclass classification problems in general. We leveraged the extensive corpus of ```multi-class labeled Amazon data to apply sentiment analysis```.

### Objective:
     Given a text book review, predict one of the three (positive, neutral, negative) sentiment classes.

This project was completed using Jupyter Notebook and Python with Pandas, NumPy, Matplotlib, Scikit-Learn, nltk, spacy and textblob.

## `Streamlit Deployment` [Deployed Link](https://share.streamlit.io/uttej2001/multiclass-text-sentiment-analysis/main/app.py)


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

First, I processed the data of review column with following steps:

    1. Lower Casing and Removed punctuations by defining a new function.
    
    2. With nltk, I imported stop words and removed them from data cleaned so far, like-i,me,myself,himself,she etc.
    
    3. I removed frequent words from data by defining a new function.
    
    4. Then I stemmarised the data, which means to presenting the summary of generated data in an easily comprehensible
       and informative manner.
       
    5. I lemmatizd the data, which is the process of grouping together the inflected forms of a word so they can be analysed as
       a single item, identified by the words lemma
      
    6. Then I removed urls and tags from the data.
    
    7. After all the above data processing steps I calculated the polarity and subjective of each review with analysis labels 
      (positive - greater than 0, negative- less than 0 and neutral- equals to 0)
      
        Positive    30296
        Negative     8659
        Neutral      3045
        
    8. I started data modeling, divided the data 33600 training and 8400 testing 
    
    9. After vectorising, I got Accuracy score on training dataset = 0.9995535714285714 and Accuracy on 
      Test dataset =0.9942857142857143
      
    10. I performed prediction on test data and got F1 score= 0.9871397964692025
    
    
![image](https://user-images.githubusercontent.com/72940291/148496085-363e4c29-b46e-4242-a227-c1dfa9f9e1e3.png)
    
  ### Results:
![image](https://user-images.githubusercontent.com/72940291/148496264-c40e2ac4-54d7-4267-af29-43d72a259682.png)

 ### Conclusion:       
   This complete notebook represents an exploratory data analysis (EDA) process to put together the patterns related to the Reviews. I have dealt with the reviews data `"multi-class labeled Amazon data to apply sentiment analysis"` and performed sentimental analysis using the textblob to calculate the (positive, neutral, negative) polarity values.  The purpose of this study is to analyze the dataset and obtain important insights form it. After performing the Data preprocessing tasks, I have developed an model using the `KNN Algorithm` for calculating the accuracy of the model. Further, using the `streamlit web application` I have deployed my model to make it available to see my work.
