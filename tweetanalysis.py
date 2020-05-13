import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

dataset = pd.read_csv('Tweets.csv', encoding = "ISO-8859-1")
dataset.info()
dataset["negativereason_confidence"].fillna(0, inplace=True)
print(dataset)

X = dataset.iloc[:5000, 4].values
y = dataset.iloc[:5000, 1].values



print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)


print(y)


import nltk
from nltk.tokenize import word_tokenize
reviews = dataset["text"].str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(reviews)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]

from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(vocabulary, y, test_size=.4, random_state=42)

print(X_train[0])

#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer()
#train_vectors = vectorizer.fit_transform(X_train)
#test_vectors = vectorizer.transform(X_test)
#print(train_vectors.shape, test_vectors.shape)
#
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(train_vectors, y_train)
#
#from  sklearn.metrics  import accuracy_score
#predicted = clf.predict(test_vectors)
#print(accuracy_score(y_test,predicted)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocabulary), 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=16,
        validation_split=0.1, verbose=1, shuffle=True)
