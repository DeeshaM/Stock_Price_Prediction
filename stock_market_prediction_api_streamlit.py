import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

#Libraries for nse and bse data collection
from nsepy import get_history
from nsetools import Nse
import yfinance as yf

#libraires for visualisation
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#Libraries for model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

#Libraries for tweet extraction and model building
import snscrape.modules.twitter as twitterScraper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Text Pre-processing
import re
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter
stop_words=stopwords.words('english')
# nlp = spacy.load('en_core_web_sm')

#-----------------------------------------------------------------------------------------
#Getting list of stocks from nse
@st.cache
def nsestocklist():
    nse = Nse()
    all_stock_codes = nse.get_stock_codes()
    list_of_stocks = list(all_stock_codes.keys())[1:]
    stock_list = ['Select stock','SBIN', 'INFY', 'DMART'] #Intentionally added at the begining
    stock_list.extend(list_of_stocks)
    return stock_list, all_stock_codes

#Downlaod NSE data for specific company

@st.cache
def load_data(comp_name):
    data = yf.download(comp_name, "2016-1-1", date.today())
    return data
#-----------------------------------------------------------------------------------------
#52 week high Price Calculation
@st.cache
def highprice(df):
    start_date = date.today() - timedelta(365)
    year_data = df["High"].loc[(df.index).date >= start_date]
    high_price = max(year_data)
    return high_price

#52 week Low Price Calculation
@st.cache
def lowprice(df):
    start_date = date.today() - timedelta(365)
    year_data = df["Low"].loc[(df.index).date >= start_date]
    low_price = min(year_data)
    return low_price

@st.cache
def stockinformation(selected_stock):
    stock_info = yf.Ticker(selected_stock).info
    return stock_info

#Profit/Loss percentage
@st.cache
def pctchange(df, n):
    data = df[["Close"]].iloc[-n:]
    data["Profit/Loss %"] = (data.pct_change())*100
    data.index = (data.index).date
    profit_data = data.dropna()
    profit_data.rename(columns={"Close" : "Closing Price"}, inplace=True)
    profit_data.round(2)
    return profit_data

def color_negative_red(val):
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color

#-----------------------------------------------------------------------------------------
#Data Preparation for LSTM model
@st.cache
def datapreparation(data_scaled):
    
    time_step = 30
    dataX, dataY = [], []
    
    # convert an array of values into a dataset matrix using time step
    for i in range(len(data_scaled)-time_step):
        dataX.append(data_scaled[i:(i+time_step), 0])    
        dataY.append(data_scaled[i + time_step, 0])
        
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    x_train = dataX.reshape(dataX.shape[0], dataX.shape[1] , 1)
    y_train = dataY
    
    return x_train, y_train
#-----------------------------------------------------------------------------------------
#LSTM Model Building
@st.cache
def model_building_prediction(neuron1, dropout_rate, x_train, y_train, epochs, batch_size, data_scaled, scalar, no_of_days_of_prediction):
    from numpy.random import seed
    seed(1)
    import tensorflow
    tensorflow.random.set_seed(2)

    model_lstm=Sequential()
    model_lstm.add(LSTM(neuron1,return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1,return_sequences=True))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1,return_sequences=True))
    model_lstm.add(Dropout(dropout_rate))
    model_lstm.add(LSTM(neuron1))
    model_lstm.add(Dropout(dropout_rate))

    model_lstm.add(Dense(1, activation='linear'))
    model_lstm.compile(loss='mean_squared_error',optimizer='adam')
    
    model_lstm.fit(x_train,y_train,epochs=epochs, batch_size=batch_size, verbose=0)
    
#-----------------------------Preicton Part-----------------------------------------------------
    x_data = data_scaled.copy()
    predicted_price_list = []
    time_steps = 30
    
    for _ in range(no_of_days_of_prediction):
        x_data = x_data[-time_steps:]
        x_data = x_data.reshape(1, time_steps, 1)
        predicted_price = model_lstm.predict(x_data)[0][0]
        predicted_price_list = np.append(predicted_price_list, predicted_price)
        x_data = np.append(x_data, predicted_price)
        
    forecasted_prices_list = scalar.inverse_transform((np.array(predicted_price_list)).reshape(-1,1))
    predicted_value_df = pd.DataFrame(forecasted_prices_list, index=daterange(no_of_days_of_prediction), 
                                      columns=["Predicted Closing Price"])
    predicted_value_df.index = pd.to_datetime(predicted_value_df.index, format="%Y-%m-%d")
    predicted_value_df.index = (predicted_value_df.index).date
    return predicted_value_df

#-----------------------------------------------------------------------------------------

#To generate future date list excluding saturdays, sundays and govt holidays
@st.cache
def daterange(no_of_days_of_prediction):
    datelist = []
    i = 1
    while len(datelist) < no_of_days_of_prediction:
        new_date = date.today() + timedelta(days=i)
        if new_date.strftime("%A") not in ["Saturday","Sunday"]:
            if new_date.strftime("%x") not in ["11/05/21","11/19/21"]:
                datelist.append(new_date.strftime("%Y-%m-%d"))
        i += 1
    return datelist

#Forecasted and recent values df to generate figure
@st.cache
def forecastedfigure(train_df, predicted_df):
    forecast_df = pd.concat([train_df["Close"].iloc[-100:], predicted_df["Predicted Closing Price"]], axis=1)
    forecast_df.rename(columns={"Close":"Closing Price", "Predicted Closing Price": "Predicted Future CP"}, inplace=True)
    return forecast_df

#-----------------------------------------------------------------------------------------
# #Function to extract stock tweets
# @st.cache
# def get_tweets(company_name):
#     currect_date = date.today()
#     # Creating list to append tweet data to
#     tweets = []
#     # Using TwitterSearchScraper to scrape data and append tweets to list
#     # Using enumerate to get the tweet and the index (to break at certain no of tweets)
#     for i,tweet in enumerate(twitterScraper.TwitterSearchScraper('{} until:{}'.format(company_name, currect_date- timedelta(days=3))).get_items()):
#         if i>700:
#             break
#         tweets.append([tweet.date, tweet.content])
#     tweet_df =pd.DataFrame(tweets, columns=['Datetime', 'Text'])
#     return tweet_df

#-----------------------------------------------------------------------------------------
# #Sentiment Analysis of collected tweets
# def tweet_sentiment(tweet):
#     output=[]
#     for i in range(len(tweet)):
#         #convert to string
#         review =str(tweet)
    
#         #to handle punctuations
#         review = re.sub('[^a-zA-Z]', ' ', tweet[i])
    
#          # Converting Text to Lower case
#         review = review.lower()

#         # Spliting each words - eg ['I','was','happy']
#         review = review.split()

#         # Applying Lemmitization for the words eg: Argument -> Argue - Using Spacy Library
#         review = nlp(' '.join(review))
#         review = [token.lemma_ for token in review]

#         # Removal of stop words
#         review = [word for word in review if word not in stop_words]

#         # Joining the words in sentences
#         review = ' '.join(review)
#         output.append(review)
    
#     x = pd.DataFrame(output)
#     # Create a SentimentIntensityAnalyzer object.
#     vader = SentimentIntensityAnalyzer()
#     x['score'] = [vader.polarity_scores(item) for item in output]
#     x['compound'] = [item['compound'] for item in x['score']]
#     # decide sentiment as positive, negative and neutral
#     x['sentiment'] = [ 'Positive' if i >= 0.05 else 'Negative' if i <= - 0.05 else 'Neutral' for i in x['compound']]
#     fig = x['sentiment'].value_counts().plot.pie(autopct=("%.2f%%"),figsize=(1,1))
#     #fig = x['sentiment'].value_counts().plot.pie(autopct="%.2f%%",figsize=(2,2), wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'}, )
#     return fig
    
#-----------------------------------------------------------------------------------------
#Above code deficts the user defined functions
#-----------------------------------------------------------------------------------------
#Main Function

#Page setup
st.set_page_config(
     page_title="Stock Market Prediction App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",)

st.title('''Stock Market Predictions''')
#Imgae display
st.image("image.jpg")

page_name=['NSE','BSE']
page=st.sidebar.radio("Security Exchanges", page_name)

best_parameters = {"SBIN":{"neuron1":40,"dropout_rate":0.1,"epochs":125,"batch_size":30},
                  "INFY":{"neuron1":40,"dropout_rate":0.1,"epochs":80,"batch_size":40},
                  "DMART":{"neuron1":60,"dropout_rate":0.0,"epochs":150,"batch_size":60},
                  "Others":{"neuron1":60,"dropout_rate":0.2,"epochs":250,"batch_size":40}}

if page == 'NSE': 
    stocks, dict_list = nsestocklist()
    selected_stock = st.sidebar.selectbox('Listed Stocks', stocks)
    data_df = load_data(selected_stock + ".NS")
     
    
else:
    _, dict_list = nsestocklist()
    stocks = ('Select stock','SBIN', 'INFY', 'DMART')
    selected_stock = st.sidebar.selectbox('Listed Stocks', stocks)
    data_df = load_data(selected_stock + ".BO")

if selected_stock == "Select stock":
    st.header("Select the stock from the menu")

else:
    st.header(f"{dict_list[selected_stock]}")
    st.write("___________________________________________________________")
  
    latest_close_price = data_df.iloc[-1,3]
    st.subheader("Price Summary")
    col1, col2, col3 = st.columns([1,1,1])
    col1.write(f"Closing Price ({((data_df.index[-1]).date())})")
    col1.write(round(latest_close_price,2))
    col2.write("52 Week High")
    col2.write(round(highprice(data_df),3))
    col3.write("52 Week Low")
    col3.write(round(lowprice(data_df),3))
    st.write("___________________________________________________________")
    
    st.subheader("Company Overview")
    stock_info = stockinformation(selected_stock + ".NS")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    col1.write('Market Cap')
    col1.write(stock_info['marketCap'])
    col2.write('P/E Ratio')
    col2.write(stock_info['trailingPE'])
    col3.write('ROE')
    if stock_info['returnOnEquity'] != None:
        col3.write('{:.2f}%'.format(stock_info['returnOnEquity']*100))
    else:
        col3.write("N/A")
    col4.write('Dividend Yield')
    if stock_info['dividendYield'] != None:
        col4.write('{:.2f}%'.format(stock_info['dividendYield']*100))
    else:
        col4.write("N/A")
    st.write("___________________________________________________________")
    
    st.line_chart(data_df["Close"])
    st.write("___________________________________________________________")
    
    st.subheader("Profit and Loss Summary of Last 5 Days")
    profit_data = (pctchange(data_df, 6)).style.applymap(color_negative_red)
    col1, col2, col3 = st.columns([2,3,2])
    col2.write(profit_data)
    
#     normalisation of data
    scaler=MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(pd.DataFrame(data_df.iloc[:,3]))
    
    x_train, y_train = datapreparation(data_scaled)
    
    if selected_stock not in ['SBIN', 'INFY', 'DMART']:
        selected_stock = "Others"

    neuron1, dropout_rate,epochs, batch_size = best_parameters[selected_stock].values()
    
    predicted_value = model_building_prediction(neuron1, dropout_rate, x_train, y_train, epochs, batch_size, data_scaled, scaler, 10)
    st.write("___________________________________________________________")
    
    st.subheader("Predictions")
    forecast_df = forecastedfigure(data_df, predicted_value)
    c1, c2=st.columns([1,2])
    c1.dataframe(predicted_value)
    c2.line_chart(forecast_df)
    st.write("___________________________________________________________")
    
#     # Stock Sentiment analysis
#     st.subheader('Stock Sentiment Analysis')
#     df_tweet = get_tweets(dict_list[selected_stock])
#     tweet_text = df_tweet['Text']
#     senti_fig = tweet_sentiment(tweet_text)
#     st.pyplot(senti_fig.figure)