
# coding: utf-8

# In[1]:

'''
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
from keras import backend as K
'''


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

'''
df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15, random_state = 42)
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
'''

df= pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and Labels
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
df['message'] = df['v2']
df.drop(['v1', 'v2'], axis=1, inplace=True)
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# In[3]:


app = Flask(__name__)


# In[4]:


@app.route('/')
def home():
    return render_template('home.html')


# In[5]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    LSTM_spam_model = open('LSTM_spam_model.pkl','rb')
    model = joblib.load(LSTM_spam_model)
    if request.method == 'POST':
        message = request.form['message']
        data = pd.Series([message.strip()])
        test_sequences = tok.texts_to_sequences(data)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
        my_prediction = model.predict(test_sequences_matrix, batch_size=1, verbose = 2)[0]
        my_prediction = int(my_prediction > 0.5)
    '''
    NB_spam_model = open('NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


# In[6]:


if __name__ == '__main__':
    app.run(debug=True)
