"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
from enum import unique
from inspect import Parameter
from typing import Sized
from PIL.Image import alpha_composite
from google.protobuf import message
from google.protobuf.message import Message
from pandas.core.frame import DataFrame
import streamlit as st
import joblib,os
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt 

# Data dependencies
import pandas as pd
import re
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
hashtags = []
def hashtag_extract(x):
			for i in x:
				ht= re.findall(r"#(w+)",i)
				hashtags.append(ht)
			return hashtags
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'', input_txt)
        
    return input_txt
raw['cl'] = np.vectorize(remove_pattern)(raw['message'], "@[\w]*")
raw['cl'] = raw['cl'].str.replace("[^a-zA-Z#]", " ")
raw['cl'] = raw['cl'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Main","EDA", "Model Performance"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Buidling out the "Main" page
	if selection == "Main":
		st.title("Team 10 Climate Change Tweet Classification:")
		st.image('CC.png', use_column_width=False)
		st.markdown("""
		Team 10:
		* **Casper Kruger:** Developed Streamlit App
		* **Kwanda Mazikubo:** Created Model
		* **Lucy Lushaba:** Created Model
		* **Gudani Mbedzi:** Created Presentation""")
		
	
	
	
	
	# Building out the "Information" page
	if selection == "EDA":
		st.title("(EDA), understanding the data!")
		st.subheader("Using graphs we can understand the data better, so from here we will look at the type of hashtags being used as well as what the data implies.")
		st.info("""General Information:   
		* **Pro** = 1     
		* **Anti** = -1     
		* **News** = 2     
		* **Neutral** = 0
		""")
		st.sidebar.header('Pro, Neutral, Anti and News:')
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		st.subheader("Raw Twitter data and label")
		
		st.dataframe(raw)
		st.info("""From here we can see the hashtags ... 
		""")
		st.sidebar.multiselect("Select status of each individual:",
		options = raw["sentiment"].unique())
		labels = 'Pro','Anti','News','Neutral'
		sizes = [16073, 2792, 6493, 5400]
		exlpode = (0.1, 0, 0, 0)
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=exlpode, labels= labels, autopct='%1.1f%%',
		shadow=True, startangle=90)
		ax1.axis('equal')
		st.pyplot(fig1, use_container_width=False)
		with st.expander('Explanation'):
			st.write(""" 
			The chart above shows us that from the data that we got, more than half of the group was pro in the belief of **man-made climate change** .
			Therefore it means that from every 10 poeple walking into a persons store there is a good chance half belief in **climate change** and that it is man made ! """)
			st.image('Graph.jpg', use_column_width=True)

	# Building out the predication page
	if selection == "Model Performance":
		st.header("Model Performance")
		st.subheader("From hear we can see the model and how it works ...")
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()