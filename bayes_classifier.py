import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation
import pandas as pd 

dataset = pd.read_csv('C:/Users/User/Dropbox/OneToOne/Python/Online Programming 5/WEEK 5/py-master/ML/14_naive_bayes/spam.csv')
print(dataset.shape)  # shows the dataset size 

print(dataset.head()) # shows the columns and first 5 rows

print(dataset.groupby('Category').describe()) # shows the number of hams and spams 

# convert the lebel in category into 1s and 0s and save in a new column 'MLcat'

dataset["MLcat"] = dataset['Category'].apply(lambda x: 1 if x== 'spam' else 0)


# spit the data set into test and train datasets 

X_train, X_test,Y_train,y_test  = train_test_split(dataset.Message, dataset.MLcat, test_size=0.25)

# Next we need to convert the words in message column into numbers 

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)


# create model  using the MultinomialNB Naive Bayes classifier
model = MultinomialNB()
#train the model
model.fit(X_train_count, Y_train)

# Test the model using 1 ham and 1 spam email
emails = [
    'Hey mohan, can we get together to watch football game tomorrow',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!']

emails_count = v.transform(emails)
print(model.predict(emails_count))


X_test_count = v.transform(X_test)
Score = model.score(X_test_count, y_test)
print("Accuracy is :" ,Score)
