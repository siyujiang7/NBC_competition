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
print("predicting")
data = pd.read_csv('HeadLine_Testingdata.csv')
# print("read")
# # Keeping only the neccessary columns
# data = data['text']
# print(data)
# # data = data[data.sentiment != "Neutral"]
# data = data.apply(lambda x: x.lower())
# # data = data.apply(lambda x: x.replace('to ', ''))
# # data = data.apply(lambda x: x.replace('the ', ''))
# # data = data.apply(lambda x: x.replace('a ', ''))
# # data = data.apply(lambda x: x.replace('of ', ''))
# # data = data.apply(lambda x: x.replace('from ', ''))
# # data = data.apply(lambda x: x.replace('is ', ''))
# # data = data.apply(lambda x: x.replace('are ', ''))
# # data = data.apply(lambda x: x.replace('was ', ''))
# # data = data.apply(lambda x: x.replace('were ', ''))
# # data = data.apply(lambda x: x.replace('for ', ''))
# # data = data.apply(lambda x: x.replace('that ', ''))
# # data = data.apply(lambda x: x.replace('there ', ''))
# # data = data.apply(lambda x: x.replace('it ', ''))
# # data = data.apply(lambda x: x.replace('its ', ''))
# # data = data.apply(lambda x: x.replace('they ', ''))
# # data = data.apply(lambda x: x.replace('he ', ''))
# # data = data.apply(lambda x: x.replace('she ', ''))
# # data = data.apply(lambda x: x.replace('his ', ''))
# # data = data.apply(lambda x: x.replace('her ', ''))
# # data = data.apply(lambda x: x.replace('and ', ''))
# # data = data.apply(lambda x: x.replace('up ', ''))
# # data = data.apply(lambda x: x.replace('on ', ''))
# # data = data.apply(lambda x: x.replace('in ', ''))
# # data = data.apply(lambda x: x.replace('at ', ''))
# # data = data.apply(lambda x: x.replace('be ', ''))
# # data = data.apply(lambda x: x.replace('been ', ''))
# # data = data.apply(lambda x: x.replace('can ', ''))
# # data = data.apply(lambda x: x.replace('an ', ''))
# # data = data.apply(lambda x: x.replace('will ', ''))
# # data = data.apply(lambda x: x.replace('you ', ''))
# # data = data.apply(lambda x: x.replace('your ', ''))
# data = data.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
# max_fatures = 2000
# tokenizer = Tokenizer(num_words=max_fatures, split=' ')
# tokenizer.fit_on_texts(data.values)
# X = tokenizer.texts_to_sequences(data.values)
# # vocal_size = len(X)
# X = pad_sequences(X)
# model = load_model('weights2.h5')
# output = model.predict(X)
# prediction = np.zeros((len(output),2))
# for i in range(0, len(prediction)):
# 	index = 0
# 	max_p = output[i][0]
# 	for j in range(0,len(output[i])):
# 		if (output[i][j] > max_p):
# 			index = j
# 			max_p = output[i][j]
# 	prediction[i][0] = i
# 	prediction[i][1] = index
# np.savetxt("predict2.csv", prediction, header= "id,sentiment", fmt='%i',delimiter=",")
