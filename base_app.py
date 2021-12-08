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
import nltk
from pandas.core.frame import DataFrame
import streamlit as st
import joblib,os
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
# Data dependencies
import pandas as pd
import re
import numpy as np
import altair as alt
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import boxcox, zscore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures

# Vectorizer
news_vectorizer = open("tfidvec_.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def hashtag_extract(tweet):  
    """Helper function to extract hashtags"""
    # creating a empty list for storage where we will keep our Hashtags later
    hashtags = []
    
    # Going through each tweet and looking for each hashtag and appending the Hashtags in our empty list hashtags
    for i in tweet:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    
    # finding the sum of the elements in the list hashtag
    hashtags = sum(hashtags, [])
    
    # creating a dictionary with tokens from the list hashtags into a dictionary, where the keys are the frequency and the values is the frequency
    frequency = nltk.FreqDist(hashtags)
    
    # creating a dataframe from the dictionary to keep track of the word and the frequency
    hashtag_df = pd.DataFrame({'hashtag': list(frequency.keys()),
                           'count': list(frequency.values())})
    
    # method is used to get n largest values from a dataframe 
    hashtag_df = hashtag_df.nlargest(25, columns="count")

    return hashtag_df


#Extracting the hashtags for the pro sentiment tweets 
pro = hashtag_extract(raw['message'][raw['sentiment'] == 1])

#Extracting the hashtags for the Anti sentiment tweets
anti = hashtag_extract(raw['message'][raw['sentiment'] == -1])

#Extracting the hashtags for the Neutral sentiment tweets
neutral = hashtag_extract(raw['message'][raw['sentiment'] == 0])

#Extracting the hashtags for the News sentiment tweets
news = hashtag_extract(raw['message'][raw['sentiment'] == 2])


#creating a dataframe with all the hashtags and a count for each sentiment
df_hashtags = pro.merge(anti,on='hashtag',suffixes=('_pro', '_anti'), how = 'outer').merge(neutral,on='hashtag', how = 'outer').merge(news,on='hashtag', suffixes = ('_neutral', '_news'), how = 'outer')


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i,'', input_txt)
        
    return input_txt
raw['cl'] = np.vectorize(remove_pattern)(raw['message'], "@[\w]*")
raw['cl'] = raw['cl'].str.replace("[^a-zA-Z#]", " ")
raw['cl'] = raw['cl'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
def lowercase(text):
    text = text.lower() 
    return text
raw['message'] = raw['message'].apply(lowercase)
st.set_page_config( layout="wide")

def camel_case_split(identifier):
    
    matches = re.finditer(
        r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
        identifier
    )
    return "  ".join([m.group(0) for m in matches])

def mentions_extractor(tweet):
    
    """function to extract mentions from the tweets"""
    mentions = re.findall(r'@([a-zA-Z0-9_]{1}[a-zA-Z0-9_]{0,14})', tweet)
  
    return mentions

raw['mentions'] = raw['message'].apply(mentions_extractor)

def urls_extractor(tweet):
    
    """function to extract urls from the tweets"""
    urls = re.findall(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', tweet)
  
    return urls
raw['urls'] = raw['message'].apply(urls_extractor)

def lowercase(text):
    text = text.lower() # making text to be lowercase
    return text
raw['message'] = raw['message'].apply(lowercase)

def lookup_dict(text, dictionary):
    
    for word in text.split(): 
        
        if word.lower() in dictionary:
            
            if word.lower() in text.split():
                
                text = text.replace(word, dictionary[word.lower()]) 
    return text

st.dataframe(raw)



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
		st.image('Images/CC.png', use_column_width=False)
		st.markdown("""
		Team 10:
		* **Casper Kruger:** Developed Streamlit App
		* **Kwanda Mazikubo:** Created Model
		* **Lucy Lushaba:** Created Model
		* **Gudani Mbedzi:** Created Presentation""")
		
	
	
	
	
	# Building out the "Information" page
	if selection == "EDA":
		st.title("(EDA), understanding the data!")
		st.image('Images/eda.jpeg', use_column_width=False)
		st.subheader("Using graphs we can understand the data better, so from here we will look at the type of hashtags being used as well as what the data implies:")
		st.info("""General Information:   
		* **Pro** = 1     
		* **Anti** = -1     
		* **News** = 2     
		* **Neutral** = 0
		""")
		st.sidebar.header('Pro, Neutral, Anti and News:')
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		st.subheader("Raw hashtag data and label")
		
		st.dataframe(df_hashtags)
		st.info("""Above we can see all the hashtags that were used, as well as how many times each hashtag correlates to the specific group. 
		""")

		chart_select = st.sidebar.selectbox(
			label = 'Select graph for most used hashtags',
			options = ['Pro', 'Anti', 'News', 'Neutral']
		)
		st.header('Hashtags correlating to each persons view on climate change!!!')
		if chart_select == 'Pro':
			df = pd.DataFrame(data= df_hashtags, columns=['hashtag','count_pro'])
			c = alt.Chart(df).mark_bar().encode(x = 'count_pro' , y = 'hashtag')
			st.altair_chart(c, use_container_width=True)

		if chart_select == 'Anti':
			ds = pd.DataFrame(data= df_hashtags, columns=['hashtag','count_anti'])
			a = alt.Chart(ds).mark_bar().encode(x = 'count_anti' , y = 'hashtag')
			st.altair_chart(a, use_container_width=True)

		if chart_select == 'News':
			dd = pd.DataFrame(data= df_hashtags, columns=['hashtag','count_neutral'])
			b = alt.Chart(dd).mark_bar().encode(x = 'count_neutral' , y = 'hashtag')
			st.altair_chart(b, use_container_width=True)


		if chart_select == 'Neutral':
			da = pd.DataFrame(data= df_hashtags, columns=['hashtag','count_news'])
			d = alt.Chart(da).mark_bar().encode(x = 'count_news' , y = 'hashtag')
			st.altair_chart(d, use_container_width=True)
		with st.expander('What we see:'):
			st.write("""With this we can select to see what the graph should show, so that we can see in each group of **Pro**, **Anti**, 
			**News**, and **Neutral** what the most used hashtags are.
			""")
			st.image('Images/Hashtag.png', use_column_width=False)
		
		

		st.header('***How the belief of climate change looks like for the group:***')
		labels = 'Pro','Anti','News','Neutral'
		sizes = [8530, 1296, 3640, 2353]
		exlpode = (0.1, 0, 0, 0)
		width = st.sidebar.slider('plot width', 0.2, 5., 6.)
		height = st.sidebar.slider('plot height', 0.2, 2.5, 6.)
		fig1, ax1 = plt.subplots(figsize=(width, height))
		
		ax1.pie(sizes,explode=exlpode, labels= labels, autopct='%1.1f%%',
		shadow=True, startangle=90)
		ax1.axis('equal')
		st.pyplot(fig1 , use_container_width=False)

		with st.expander('Explanation'):
			st.write(""" 
			The chart above shows us that from the data that we got, more than half of the group was pro in the belief of **man-made climate change** .
			Therefore we can assume than majority of any business has half of it's client's believing that **climate change is man-made** ! """)
			st.image('Images/Graph.jpg', use_column_width=False)
		st.header("""Here we can see the freqeunt words used in each group:
		Top left: News 
		Top right: Pro 
		Bottom left: Anti 
		Bottom right: Neutral """)
		st.image('Images/Words.png', use_column_width=False)
		st.markdown("""
		Observations:
		* The most frequent words across all 4 classes are **climatechange** and **rt**. The word **rt(retweet)** indicates that there is a massive amount of information that resonate with the users and it is shared with a broader audience.
		* The word **trump** is also present in all 4 classes, this was anticapated considering his sentiment and overall presence on social media.
		* The news climate change class has the **http**, it represents the urls addresses that the news outlets share to direct users to articles.
		* The pro climate change class has words **deniers**, **think**, **believe**. The word **deniers** might be refering to the people whom they believe to be indenial of climate change and think, believe might be refering to words they use to convince non-believers about climate change.
		*  The anti climate change class has words **hoax**, **scam**, **fake** which might indicate that they believe that climate change/global warming is a hoax.""")

	# Building out the predication page
	if selection == "Model Performance":
		st.title("Model Performance:")
		st.image('Images/perform.jpg', use_column_width=False)
		st.subheader("From hear we can see the Different models we used as well as how they performed:")
		st.info("Logistic regression works by measuring the relationship between the target variable (what we want to predict) and one or more predictor. It does this by estimating the probabilities with the help of its underlying logistic function. ")
		st.subheader('It is represented by the equation:')
		st.latex(r'''y = \left(\frac{e^{b0+b1*x}}{1+e^{b0+b1*x}}\right)
		
		''')
		# Creating a text box for user input
		st.info("""General Information: 

		 y = The ouput of the function 

		 b0 = The bias or intercept  

		 e = The base of the natural logarithms 

		 b1 = The coefficient for the input

		 x = The predictor variable
		""")

		st.header('Each model that we use, with their positives and negatives:')

		col1, col2, col3, col4 = st.columns(4)

		with col1:
			st.header('Logistic Regression')
			st.info('''
			Positives:
			* Is easier to impliment, interpret, and very efficient to train .
			* It makes no assumptionsof classes in feature space .
			* It can easily extend to multiple classes(multinomial regression) and a natural probabilistic view of class predictions. 
			
			Negatives:
			* If the number of observations is lesser than the number of features, Logistic Regression should not be used, otherwise, it may lead to overfitting.
			* It constructs linear boundaries.
			* The major limitation is the assumption of linearity between the dependent variable and the independent variables.
			''')

		with col2:
			st.header('Random Forest Classifier')
			st.info('''
			Positives:
			* Can be used to solve both classification as well as regression problems.
			* Works well with both categorical and continuous variables.
			* It creates as many trees on the subset of the data and combines the output of all the trees. In this way it reduces overfitting problem in decision trees and also reduces the variance and therefore improves the accuracy.
			
			Negatives:
			* Complexity is increase because it creates multiple trees and combines their outputs
			* Training period takes longer because it takes a majority of the votes from the trees
			''')

		with col3:
			st.header('Linear Support Vector Classification')
			st.info('''
			Positives:
			* Works relatively well when there is a clear margin of separation between classes.
			* Is more effective in high dimensional spaces.
			* Is effective in cases where the number of dimensions is greater than the number of samples. 
			
			Negatives:
			* Algorithm is not suitable for large data sets.
			* Does not perform very well when the data set has more noise i.e. target classes are overlapping.
			* In cases where the number of features for each data point exceeds the number of training data samples, the SVC will underperform.
			''')

		with col4:
			st.header('XGBClassifier')
			st.info('''
			Positives:
			* Less feature engineering required
			* Fast to interpret
			* Outliers have minimal impact.
			
			Negatives:
			* Harder to tune as there are too many hyperparameters.
			* Difficult interpretation , visualization tough
			* Overfitting possible if parameters not tuned properly.
			''')
		st.header('How each model performed on an f1-score:')
		st. latex(r'''F1 score = 2 \left(\frac{Precision * Recall}{Pecision + Recall}\right)''')
		st.info('F1 score is the measure of a tests accuracy or in this case our models accuracy. It is calculated as shown above, where the precision is the number of true positive results is devided by the number of all positive results.')
		st.image('Images/f1.jfif', use_column_width=False)

		tweet_text = st.text_area("Enter Text","Type here")
		Logistic = ("lr_model.pkl")
		Random = ("rfc_model.pkl")
		Linear = ('lsvc_model.pkl')
		XGBmodel = ('xgb_model.pkl')

		original_list = [ Logistic, Random, Linear, XGBmodel ]
		st.info('''
		Models:
		* **LogisticRegression Model** = lr_model.pkl
		* **RandomForestClassifier Model** = rfc_model.pkl
		* **LinearSupportVectorClassifier Model** = lsvc_model.pkl
		* **XGBoosterClassifier** = xgb_model.pkl
		''')
		result = st.selectbox('Select the model you want to use:', original_list)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(result),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		st.info(''' 
		Categorized as:
		* **Pro** = 1
		* **Anti** = -1
		* **News** = 2
		* **Neutral** = 0
		
		''')
	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()