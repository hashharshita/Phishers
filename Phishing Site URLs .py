#!/usr/bin/env python
# coding: utf-8

# ###  Importing some useful libraries

# In[3]:


import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix

import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
get_ipython().run_line_magic('matplotlib', 'inline')
# It sets the backend of matplotlib to the 'inline' backend:
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')


# * **Loading the main dataset.**

# In[4]:


phish_data = pd.read_csv(r'C:\Users\harsh\Desktop\Phishing_Site_Prediction-master\phishing_site_urls.csv')


# #### download dataset from my **Kaggle**  <a href='https://www.kaggle.com/taruntiwarihp/phishing-site-urls'>here</a>

# In[5]:


phish_data.head()


# In[6]:


phish_data.tail()


# In[7]:


phish_data.info()


# * **About dataset**
# * Data is containg 5,49,346 unique entries.
# * There are two columns.
# * Label column is prediction col which has 2 categories 
#     A. Good - which means the urls is not containing malicious stuff and **this site is not a Phishing Site.**
#     B. Bad - which means the urls contains malicious stuffs and **this site isa Phishing Site.**
# * There is no missing value in the dataset.

# In[8]:


phish_data.isnull().sum() # there is no missing values


# * **Since it is classification problems so let's see the classes are balanced or imbalances**

# In[9]:


#create a dataframe of classes counts
label_counts = pd.DataFrame(phish_data.Label.value_counts())


# In[10]:


#visualizing target_col
sns.set_style('darkgrid')
sns.barplot(label_counts.index,label_counts.Label)


# ### Preprocessing

# * **Now that we have the data, we have to vectorize our URLs. I used CountVectorizer and gather words using tokenizer, since there are words in urls that are more important than other words e.g ‘virus’, ‘.exe’ ,’.dat’ etc. Lets convert the URLs into a vector form.**

# #### RegexpTokenizer
# * A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens.

# In[11]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[12]:


phish_data.URL[0]


# In[13]:


# this will be pull letter which matches to expression
tokenizer.tokenize(phish_data.URL[0]) # using first row


# In[14]:


print('Getting words tokenized ...')

phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows


# In[15]:


phish_data.sample(5)


# #### SnowballStemmer
# * Snowball is a small string processing language, gives root words

# In[16]:


stemmer = SnowballStemmer("english") # choose a language


# In[17]:


print('Getting words stemmed ...')
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])


# In[18]:


phish_data.sample(5)


# In[19]:


print('Getting joiningwords ...')
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))


# In[20]:


phish_data.sample(5)


# ### Visualization 
# **1. Visualize some important keys using word cloud**

# In[21]:


#sliceing classes
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']


# In[22]:


bad_sites.head()


# In[23]:


good_sites.head()


# * create a function to visualize the important keys from url 

# In[24]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  


# In[25]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[26]:


common_text = str(data)
common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)


# In[27]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[28]:


common_text = str(data)
common_mask = np.array(Image.open('comment.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in bad urls', title_size=15)


# Download more various type of images <a href='https://github.com/taruntiwarihp/raw_images/tree/master/Words%20cloud%20images'>here</a>

# **2. Visualize internal links, it will shows all redirect links.** 

# #### Scrape any website
# * First, setting up the Chrome webdriver so we can scrape dynamic web pages.

# #### Chrome webdriver
# * WebDriver tool use for automated testing of webapps across many browsers. It provides capabilities for navigating to web pages, user input and more

# In[29]:


browser = webdriver.Chrome(executable_path=r"C:\Users\Asus\Downloads\chromedriver_win32\chromedriver")


# **You can download chromedriver.exe from my github <a href='https://github.com/taruntiwarihp/dataSets/blob/master/chromedriver_win32.zip'>here</a>**

# * After set up the Chrome driver create two lists.
# * First list named list_urls holds all the pages you’d like to scrape.
# * Second, create an empty list where you’ll append links from each page.
# 

# In[ ]:


list_urls = ['https://www.republicworld.com/topics/filmy4wap'] #here i take phishing sites 
links_with_text = []


# * I took some phishing site to see were the hackers redirect(on different link) us.
# * Use the BeautifulSoup library to extract only relevant hyperlinks for Google, i.e. links only with '<'a'>' tags with href attributes. 

# #### BeautifulSoup
# * It is use for getting data out of HTML, XML, and other markup languages. 

# In[ ]:


for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])


# #### Turn the URL’s into a Dataframe
# * After you get the list of your websites with hyperlinks turn them into a Pandas DataFrame with columns “from” (URL where the link resides) and “to” (link destination URL)

# In[ ]:


df = pd.DataFrame(links_with_text, columns=["from", "to"])


# In[ ]:


df.head()


# #### Step 3: Draw a graph
# * Finally, use the aforementioned DataFrame to **visualize an internal link structure by feeding it to the Networkx method from_pandas_edgelist first** and draw it by calling nx.draw

# In[ ]:


GA = nx.from_pandas_edgelist(df, source="from", target="to")
nx.draw(GA, with_labels=False)


# ### Creating Model

# ![image.png](attachment:image.png)
# 
# * Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
# 

# #### CountVectorizer
# * CountVectorizer is used to transform a corpora of text to a vector of term / token counts.

# In[ ]:


#create cv object
cv = CountVectorizer()


# In[ ]:


#help(CountVectorizer())


# In[ ]:


feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed


# In[ ]:


feature[:5].toarray() # convert sparse matrix into array to print transformed features


# #### * Spliting the data 

# In[ ]:


trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)


# ### LogisticRegression
# 

# In[ ]:


# create lr object
lr = LogisticRegression()


# In[ ]:


lr.fit(trainX,trainY)


# In[ ]:


lr.score(testX,testY)


# In[ ]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# *** So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression**

# In[ ]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())


# In[ ]:


trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)


# In[ ]:


pipeline_ls.fit(trainX,trainY)


# In[ ]:


pipeline_ls.score(testX,testY) 


# In[ ]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[ ]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[ ]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# ***That’s it. See, it's that simple yet so effective. We get an accuracy of 98%. That’s a very high value for a machine to be able to detect a malicious URL with. Want to test some links to see if the model gives good predictions? Sure. Let's do it**

# * Bad links => this are phishing sites
# yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php
# fazan-pacir.rs/temp/libraries/ipad
# www.tubemoviez.exe
# svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt
# 
# * Good links => this are not phishing sites
# www.youtube.com/
# youtube.com/watch?v=qI0TQJI3vdU
# www.retailhellunderground.com/
# restorevisioncenters.com/html/technology.html

# In[ ]:


predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)


# https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset

# In[ ]:


predict_bad = ['https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.predict(predict_bad)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




