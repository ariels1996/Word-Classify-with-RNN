from keras.layers import Dense, Dropout, LSTM, Embedding, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import pandas as pd
import numpy as np

input_file = 'data1.csv'

#make train_dataset & test data set
def load_data(test_split = 0.3):
    print ('Loading data...')
    df = pd.read_csv(input_file)

    #change item to str(char-seq-list)
    df['pattern'] = df['pattern'].apply(lambda x: str(x))
    
    X_data = text_encoder(df['pattern'])
    # print(len(X_data))

    # #randomize dataframe
    # df = df.reindex(np.random.permutation(df.index))
    
    #set train size
    train_size = int(len(df) * (1 - test_split))

    #make train set
    X_train = X_data[:train_size]
    y_train = np.array(df['isVulnerable'].values[:train_size])
    #make test set
    X_test = X_data[train_size:]
    y_test = np.array(df['isVulnerable'].values[train_size:])

    return X_train, y_train, X_test, y_test

def text_encoder(data):
    # High frequency 200 char in string (Tokenizer)
    tokenizer = Tokenizer(200)
    # make index of character
    tokenizer.fit_on_texts(data)
    # char to int
    sequences = tokenizer.texts_to_sequences(data)
    # show how many index in sequences
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    new_data = pad_sequences(sequences, maxlen= 30, padding='post')
    print (new_data)
    return new_data

def create_model(input_length):
   
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 200, output_dim = 50, input_length = input_length))
    model.add(LSTM(128, activation='sigmoid', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


X_train, y_train, X_test, y_test = load_data()
print(len(X_train[0]))
model = create_model(len(X_train[0]))
print(y_train)
print ('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split = 0.1, verbose = 1)

score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)

#save the model
model.save('ReDos_LSTM.h5')

#predict the test file
y_pred = model.predict(X_test)
