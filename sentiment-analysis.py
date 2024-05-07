import os
os.getcwd()
os.chdir("C:/Users/Albin/Documents/GitHub/sentiment-analysis")
os.getcwd()
import pandas as pd
df=pd.read_csv("customer_reviews.csv")


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sa=SentimentIntensityAnalyzer()

#checking sentiments for first feedback
sa.polarity_scores(df.iloc[20,1])
print(df.iloc[20,1])
#checking sentiments of text
df["score"]=df["text"].apply(lambda x:sa.polarity_scores(x))
#extracting compound score
df["compound_score"]=df["score"].apply(lambda x:x["compound"])

import numpy as np
df["positive_negative"]=df["compound_score"].apply(lambda x:np.where(x>0.5,"Positive","Negative"))
#counting of negative and positive feedback
df["positive_negative"].value_counts()
positive_data=df.query("positive_negative=='Positive'")