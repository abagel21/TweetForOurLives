import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/dreaddit-test.csv")
a = df["text"]
b = df["label"]
df = pd.concat([a, b], axis=1)
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in df.iterrows():
    row[0] = row[0].replace('rt',' ')
max_features = 2000
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)

pd.DataFrame(X).to_csv("test.csv")