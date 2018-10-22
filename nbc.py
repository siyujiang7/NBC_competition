import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')


def train():
    stemmer = PorterStemmer()
    data = pd.read_csv('HeadLine_Trainingdata.csv')

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    data = data.as_matrix()
    # Keeping only the neccessary columns
    X = []
    vocab = []
    prior = np.zeros(5)
    for line in data:
        if(line[2] == 0):
            prior[0] += 1

        if (line[2] == 1):
            prior[1] += 1

        if (line[2] == 2):
            prior[2] += 1

        if (line[2] == 3):
            prior[3] += 1

        if (line[2] == 4):
            prior[4] += 1

        sentence = line[1]
        tokens = word_tokenize(sentence)
        tokens_pos = pos_tag(tokens)
        list = []
        for word in tokens_pos:
            if ((word[1] != 'DT') & (word[0] not in stop_words) & (word[0] != 'is') & (word[0] != 'are') & (word[0] != 'be') & (word[0] != 'was') & (word[0] != 'were') & (word[0] != 'been')):
                w = stemmer.stem(word[0])
                list.append(w)
                if(w not in vocab):
                    vocab.append(w)
        X.append((line[0], list, line[2]))
    # print(vocab.index("second"))
    # print(vocab)
    word_freq = np.zeros((len(vocab),5))

    for entry in X:
        tokens = entry[1]
        for word in tokens:
            index = vocab.index(word)
            word_freq[index][entry[2]]+=1

    print(prior)
    print(word_freq)
    length = len(X)
    print(length)
    return (prior, word_freq, vocab, length)

def predict(stats):
    prior = stats[0]
    word_freq = stats[1]
    vocab = stats[2]
    length = stats[3]
    most_likely  = 0
    max_occur = prior[0]
    for i  in range(0,5):
        if(prior[i] > max_occur):
            most_likely = i
            max_occur = prior[i]
    print(most_likely)
    # V = 0
    # for j in range(0,5):
    #     for i in range(0,len(word_freq)):
    #         V += word_freq[i][j]
    # print(V)
    stemmer = PorterStemmer()
    data = pd.read_csv('HeadLine_Testingdata.csv')
    print("read")

    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    data = data.as_matrix()
    X = []
    for line in data:
        sentence = line[1]
        tokens = word_tokenize(sentence)
        tokens_pos = pos_tag(tokens)
        list = []
        for word in tokens_pos:
            if ((word[1] != 'DT') & (word[0] not in stop_words) & (word[0] != 'is') & (word[0] != 'are') & (word[0] != 'be') & (word[0] != 'was') & (word[0] != 'were') & (word[0] != 'been')):
                w = stemmer.stem(word[0])
                list.append(w)
        X.append((line[0], list))

    prediction = np.zeros(len(X))
    z = 0
    for entry in X:
        tokens = entry[1]
        likehood = np.zeros(5)
        for i in range(0,len(likehood)):
            product = 1
            p = prior[i]
            for word in tokens:
                freq = 0
                if(word in vocab):
                    freq = word_freq[vocab.index(word)][i]
                prob_w = (freq)/(p)
                product *= prob_w
            likehood[i] = product*p/length
        index = most_likely
        max = likehood[most_likely]
        for j in range(0,len(likehood)):
            if(likehood[j] > max):
                index = j
                max = likehood[j]
        prediction[z] = index
        z+=1

    report = np.zeros((len(prediction),2))
    for i  in range(0,len(report)):
        report[i][0] = i
        report[i][1] = prediction[i]

    np.savetxt("nbcpredicttest2.csv", report, header="id,sentiment", fmt='%i', delimiter=",")
    # count = 0
    # for i in  range(0,len(data)):
    #     if (prediction[i] == data[i][2]):
    #         count+=1
    #
    # print(count/len(data))


stats = train()
predict(stats)
