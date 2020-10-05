import pandas as pd 
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import datetime

def processDF(df) :
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    for idx,row in df.iterrows():
        row[0] = row[0].replace('rt',' ')
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    return X

# set up to run on gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPU config set")

# import and split train and test data
train = pd.read_csv("data/dreaddit-train.csv")
test = pd.read_csv("data/dreaddit-test.csv")
df = train.append(test, ignore_index=True)
df = pd.concat([df['text'], df['label']], axis=1)
df_train = processDF(df)
df_test = df['label']
X_train, X_test, y_train, y_test = train_test_split(df_train, df_test, test_size=0.2, random_state=42)

# importing necessary classes
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow_addons.optimizers import AdamW

# preparing callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience='50')

# preparing optimizers
opti = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
decayedOpti = AdamW(weight_decay = 0.00001, learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamW')

# preparing initializer and regularizer
initializer = GlorotUniform()
regularizer = L2(l2=0.1)

#hyperparameters
EPOCHS = 50
max_features=2000
embed_dim=64
lstm_out=196
batch_size = 128


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, SpatialDropout1D, Embedding, LSTM

# activation='tanh', recurrent_activation='sigmoid', use_bias=True,
#    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
#    bias_initializer='zeros', 
def make_model():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length = X_train.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(
    256, dropout = 0.7, recurrent_dropout=0.7))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opti)
    return model

model = make_model()
print(model.summary())
model.fit(X_train, y_train, epochs= EPOCHS, validation_data=(X_test, y_test), batch_size=batch_size, callbacks=[tensorboard_callback, early_stop])

# evaluate and save model
from sklearn.metrics import mean_absolute_error, explained_variance_score
pred = model.predict(X_test)
print("Validation Scores")
print("mae=" + mean_absolute_error(y_test, pred))
print("explained variance="+ explained_variance_score(y_test, pred))

model.save(f'saved_models/{model_dir}')

