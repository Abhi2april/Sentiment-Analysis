import streamlit as st  
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'POLARITY':sentiment.polarity,'SUBJECTIVITY':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['Parameter','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	positive_list = []
	negative_list = []
	neutral_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			positive_list.append(i)
			positive_list.append(res)

		elif res <= -0.1:
			negative_list.append(i)
			negative_list.append(res)
		else:
			neutral_list.append(i)

	result = {'POSITIVE':positive_list,'NEGATIVE':negative_list,'NEUTRAL':neutral_list}
	return result 

    
def main():
    st.title('Multi Class Text Sentiment Analysis Using Amazon Review Data')
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color: white; text-align: center;">Analysis using the KNN Algorithm </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter the Text Here")
            submit_button = st.form_submit_button(label='Analyze')
    
        col1,col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("The Output : ")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)
                
                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")
                
                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)
                
            with col2:
                st.info("Token Sentiment")
                
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)
    
    else:
        st.subheader("About the Analysis:")
        """
                Predicting a review's numeric rating based on the textual review is a quintessential multiclass 
        text classification problem and an interesting research topic in ```Natural Language Processing```. 
        New advances in NLP including the development of Glove, and Word2Vec, have increased the range of 
        approaches available to address this question, and insights gained on this problem can generalize across 
        sentiment analysis and NLP multiclass classification problems in general. We leveraged the extensive corpus 
        of ```multi-class labeled Amazon data to apply sentiment analysis```.
        
        > ```# Objective```: Given a text book review, predict one of the three ```(positive, neutral, negative)``` sentiment classes.
        """

if __name__ == '__main__':
	main()