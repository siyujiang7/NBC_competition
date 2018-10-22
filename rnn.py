# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import load_model
from keras.callbacks import Callback
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

data = pd.read_csv('HeadLine_Trainingdata.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

# data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())

# data['text'] = data['text'].apply(lambda x: x.replace('to ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('the ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('a ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('of ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('from ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('is ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('are ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('was ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('were ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('for ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('that ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('there ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('it ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('its ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('they ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('he ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('she ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('his ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('her ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('and ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('up ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('on ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('in ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('at ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('be ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('been ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('can ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('an ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('will ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('you ', ''))
# data['text'] = data['text'].apply(lambda x: x.replace('your ', ''))


data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))


print(data['text'])

print(data[ data['sentiment'] == 0].size)
print(data[ data['sentiment'] == 1].size)
print(data[ data['sentiment'] == 2].size)
print(data[ data['sentiment'] == 3].size)
print(data[ data['sentiment'] == 4].size)
# for idx,row in data.iterrows():
#     row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
# vocal_size = len(X)
X = pad_sequences(X)
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 10
model.fit(X_train, Y_train, epochs =100, callbacks=[TestCallback((X_test, Y_test))], batch_size=batch_size, verbose = 2)

# print(len(X_test))
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

model.save("weights2.h5")