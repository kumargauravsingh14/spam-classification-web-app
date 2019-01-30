# spam-sms-classification

In this project, we model a classifier to label a given piece of text as a `spam` or `not a spam`. Also, we created an API for the model, using Flask, the Python micro framework for building web applications. 
The model is currently being hosted on herokuapp server - [here](https://spamsmsclassifier.herokuapp.com/)

## Dataset

The classifier is trained offline with spam and non-spam messages. The trained model is deployed as a service to serve users. We have used the famous [SPAM or HAM Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) by [UCI-ML](https://archive.ics.uci.edu/ml/index.html).

The dataset consists of a file naming `data.csv` which contains one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

## Features

We have used Naive Bayes and Count Vectorizer. Later its accuracy has been improved by using SVM Classifier and TF-IDF Vectorizer. It is further improved by using LSTM classifier implemented using Keras API in Python.

## Tools

Main modules used are - Flask, Pandas, sklearn, numpy, PIL, Keras, etc.
