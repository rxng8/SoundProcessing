"""
    Author: Alex Nguyen
    Gettysburg College class of 2022
    This file get into the analysis of sound processing and some of its application

    Note: I have written every single lines
    The documentation will be written as the file goes along.

"""
# %%

class Config:
    def __init__(self, var, x):
        self.var = var
        self.x = x

# %%

import os
import sunau
_fname = '../genres/genres/blues/blues.00000.au'
data = sunau.open(_fname, 'r')

x = data.readframes(22050)

y = int.from_bytes(x, byteorder='big')

# %%

""" Importing library """

# OS, IO
from scipy.io import wavfile
import os, sys

# Sound Processing library
import librosa
from pydub import AudioSegment

# Math Library
import numpy as np

# Display library
import IPython.display as ipd
import matplotlib.pyplot as plt
%matplotlib inline
plt.interactive(True)
import librosa.display

# Data Preprocessing
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import sklearn

# Deep Learning Library
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, LSTM, Bidirectional, GRU
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import Sequence
from keras.optimizers import Adam, SGD, RMSprop

# %%
""" Test """

audio, sr = librosa.core.load(fname) # float32
# sr1, audio1 = wavfile.read(fname) # int16

# %%

audio1[:10]

# %%

# from pydub import AudioSegment

# x = AudioSegment.from_file(fname)
# x.export('out.wav', format='wav')


# %%
# import numpy as np
# import IPython.display as ipd

# ipd.Audio(np.array([i for i in range(100000)]), rate=sr1)

# %%

# ipd.Audio(audio, rate=sr)

# %%
plt.figure(figsize=(14, 5))
waveplot(audio, sr=sr)
plt.show()

# %%

X = librosa.stft(audio)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

# %%

# Dataset Separation

# No need

# %%

# Dataset and classes definition

base_dir = 'dataset/genres_converted'

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'raggae', 'rock']

# %%



# %%

""" Data Preprocessor class """

class DataPreprocessor:

    """
    Converting labels to integers
    :param labels: (1D np array) An array of associated labels with data
    :return: (1D np array) An array of integer representing the associated labels
    """
    @staticmethod
    def tokenize(labels):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(labels)
        label_matrix = np.asarray(tokenizer.texts_to_sequences(labels))
        y = np.reshape(label_matrix, label_matrix.shape[0])
        return y, tokenizer
    
    @staticmethod
    def generate_label(classes):
        label_matrix, tokenizer = DataPreprocessor.tokenize(classes)
        return to_categorical(label_matrix), tokenizer

    """
    :param fname:
    :param window_size: (Integer) the max length of axis 1 of x.
    """
    @staticmethod
    def get_spect(fname):
        y, sr = librosa.core.load(fname)
        spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
        spect = librosa.power_to_db(spect, ref=np.max)
        # Normalization
        # for i, feature in enumerate(spect):
        #     max_val = feature.max(axis=0)
        #     min_val = feature.min(axis=0)
        #     diff = max_val - min_val
        #     for j, time in enumerate(feature):
        #         spect[i][j] = (time - min_val) / diff
        return spect

    @staticmethod
    def get_mfcc(fname, n_mfcc=40):
        y, sr = librosa.core.load(fname)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        """ Feature Scaling MFCC """
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        return mfccs

    @staticmethod
    def pad_matrix(matrix, max_length):
        return np.concatenate((matrix, np.zeros(shape=(matrix.shape[0], max_length - matrix.shape[1]))), axis=1)

# %%

# Test data preprocessor:
x, _ = DataPreprocessor.generate_label(classes)



# %%
cnt = 0
for r, d, f in os.walk(base_dir):
    print(r)

    # cnt +=1
    # if cnt == 30:
    #     break

# %%

""" Data Generator Class """

class AudioDataGenerator(Sequence):
    """
    :param data_path: (String) This is the base folder data.
    :param batch_size: (int32) This is the base folder data.
    :param dim: (Tuple: (a, b, c)) 3D tuple shape of input dimension
    :param n_channels: (int32) Number of channel.
    :param n_classes: (int32) Number of classes.
    :param shuffle: (boolean) Specify whether or not you want to shuffle the data to be trained.
    """
    def __init__(self, data_path, batch_size=32, dim=(128,1308), n_channels=1,
             n_classes=10, shuffle=True):
        """
        :var self.classes:
        :var self.labels:
        :var self.fname:
        :var self.data:
        :var self.dim:
        :var self.batch_size:
        :var self.list_IDs:
        :var self.n_channels:
        :var self.n_classes:
        :var self.shuffle:
        :var self.tokenizer:
        :var self.data_path:
        """
        self.classes = []
        self.labels = []
        self.fname = []
        self.data = []

        self.data_size = 0
        self.data_shape = (None,None)
        self.data_path = data_path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_data()
        
    """
    :param data_path: (String) The actual base folder of data
    """
    def load_data(self):

        # Generate labels and convert to categorized 2D-vector
        for i, _cls in enumerate(os.listdir(self.data_path)):
            self.classes.append(_cls)
            for j, fname in enumerate(os.listdir(os.path.join(self.data_path, _cls))):
                self.fname.append(os.path.join(base_dir, _cls, fname))
                self.labels.append(_cls)
                
        print("Found {} classes in root data folder".format(len(self.classes)))
        self.labels, self.tokenizer = DataPreprocessor.generate_label(self.labels)
        self.labels = np.asarray(self.labels)
        self.data_size = self.labels.shape[0]
            
    
    def build_data_stft(self):
        print('Building data...')

        temp_data = []
        max_length = 0
        for i, fname in enumerate(self.fname):
            spect = DataPreprocessor.get_spect(fname)
            temp_data.append(spect)
            if spect.shape[1] > max_length:
                max_length = spect.shape[1]

        self.data = np.zeros(shape=(self.data_size, self.dim[0], max_length))

        for i, spect in enumerate(temp_data):
            if spect.shape[1] < max_length:
                self.data[i] = DataPreprocessor.pad_matrix(spect, max_length)
            else:
                self.data[i] = spect

        self.data = np.expand_dims(self.data, axis=3)

        print('Build Data Successful!')
        print('Found {} pieces of data and converted each piece in to shape {}'.format(self.data_size, self.dim[1:]))


        # datums = []
        # for i, fname in enumerate(self.fname):
        #     datums.append(DataPreprocessor.get_spect(fname))

        # max_length = 0
        # for i, datum in enumerate(datums):
        #     if datum.shape[1] > max_length:
        #         max_length = datum.shape[1]

        # self.data = np.zeros(shape=(self.data_size, self.dim[0], max_length))
        # for i, datum in enumerate(datums):
        #     if max_length > datum.shape[1]:
        #         datum = np.concatenate((datum, np.zeros(shape=(datum.shape[0], max_length - datum.shape[1]))), axis=1)
        #     self.data[i] = datum
            
        # print('Build Data Successful!')
        # print('Found {} pieces of data and converted each piece in to shape {}'.format(self.data_size, max_length))

    def build_data_mfcc(self):
        for fname in self.fname:
            datum = DataPreprocessor.get_mfcc(fname)
        return

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        return

    def __len__(self):
        return

    def __getitem__(self, index):
        return

# %%

""" Data Generator Class """

class AudioDataGenerator2D(Sequence):
    """
    :param data_path: (String) This is the base folder data.
    :param batch_size: (int32) This is the base folder data.
    :param dim: (Tuple: (a, b, c)) 3D tuple shape of input dimension
    :param n_channels: (int32) Number of channel.
    :param n_classes: (int32) Number of classes.
    :param shuffle: (boolean) Specify whether or not you want to shuffle the data to be trained.
    """
    def __init__(self, data_path, batch_size=32, dim=(128,1308), n_channels=1,
             n_classes=10, shuffle=True):
        """
        :var self.classes:
        :var self.labels:
        :var self.fname:
        :var self.data:
        :var self.dim:
        :var self.batch_size:
        :var self.list_IDs:
        :var self.n_channels:
        :var self.n_classes:
        :var self.shuffle:
        :var self.tokenizer:
        :var self.data_path:
        """
        self.classes = []
        self.labels = []
        self.fname = []
        self.data = []

        self.data_size = 0
        self.data_shape = (None,None)
        self.data_path = data_path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_data()
        
    """
    :param data_path: (String) The actual base folder of data
    """
    def load_data(self):

        # Generate labels and convert to categorized 2D-vector
        for i, _cls in enumerate(os.listdir(self.data_path)):
            self.classes.append(_cls)
            for j, fname in enumerate(os.listdir(os.path.join(self.data_path, _cls))):
                self.fname.append(os.path.join(base_dir, _cls, fname))
                self.labels.append(_cls)
                
        print("Found {} classes in root data folder".format(len(self.classes)))
        self.labels, self.tokenizer = DataPreprocessor.generate_label(self.labels)
        self.labels = np.asarray(self.labels)
        self.data_size = self.labels.shape[0]
            
    
    def build_data_stft(self):
        print('Building data...')

        temp_data = []
        max_length = 0
        for i, fname in enumerate(self.fname):
            spect = DataPreprocessor.get_spect(fname)
            temp_data.append(spect)
            if spect.shape[1] > max_length:
                max_length = spect.shape[1]

        self.data = np.zeros(shape=(self.data_size, self.dim[0], max_length))

        for i, spect in enumerate(temp_data):
            if spect.shape[1] < max_length:
                self.data[i] = DataPreprocessor.pad_matrix(spect, max_length)
            else:
                self.data[i] = spect

        self.data = np.expand_dims(self.data, axis=3)

        # datums = []
        # for i, fname in enumerate(self.fname):
        #     datums.append(DataPreprocessor.get_spect(fname))

        # max_length = 0
        # for i, datum in enumerate(datums):
        #     if datum.shape[1] > max_length:
        #         max_length = datum.shape[1]

        # self.data = np.zeros(shape=(self.data_size, self.dim[0], max_length))
        # for i, datum in enumerate(datums):
        #     if max_length > datum.shape[1]:
        #         datum = np.concatenate((datum, np.zeros(shape=(datum.shape[0], max_length - datum.shape[1]))), axis=1)
        #     self.data[i] = datum
            
        # print('Build Data Successful!')
        # print('Found {} pieces of data and converted each piece in to shape {}'.format(self.data_size, max_length))

        print('Build Data Successful!')
        print('Found {} pieces of data and converted each piece in to shape {}'.format(self.data_size, self.dim[1:]))

    def build_data_mfcc(self):
        for fname in self.fname:
            datum = DataPreprocessor.get_mfcc(fname)
        return

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        return

    def __len__(self):
        return

    def __getitem__(self, index):
        return


# %%

""" Build Datagenerator """

dataGen = AudioDataGenerator2D(base_dir)

# %%

dataGen.build_data_stft()

# %%

dataGen.data.shape

# %%

f = []

# %%

datum = DataPreprocessor.get_spect(dataGen.fname[0])

# %%

np.append(f, datum).reshape(128,1293)

# %%

datum.shape

# %%

# np.asarray(dataGen.data)


len(dataGen.data)

# %%

x = np.zeros(shape=(1000,128,1293))
x[0] = dataGen.data[0]

# %%

x
# %%

dataGen.build_data_stft()

# %%

dataGen.labels

# %%

""" Considering mel spectrogram of different kind of music """


ftest1 = 'dataset/genres_converted/classical/classical.00000.wav'
ftest2 = 'dataset/genres_converted/blues/blues.00000.wav'
ftest3 = 'dataset/genres_converted/hiphop/hiphop.00000.wav'
ftest4 = 'dataset/genres_converted/jazz/jazz.00000.wav'
ftest5 = 'dataset/genres_converted/metal/metal.00000.wav'



f1 = DataPreprocessor.get_spect(ftest1)
f2 = DataPreprocessor.get_spect(ftest2)
f3 = DataPreprocessor.get_spect(ftest3)
f4 = DataPreprocessor.get_spect(ftest4)
f5 = DataPreprocessor.get_spect(ftest5)

plt.figure(figsize=(7,3))
librosa.display.specshow(f1, sr=22050, x_axis='time', y_axis='hz')

plt.figure(figsize=(7,3))
librosa.display.specshow(f2, sr=22050, x_axis='time', y_axis='hz')

plt.figure(figsize=(7,3))
librosa.display.specshow(f3, sr=22050, x_axis='time', y_axis='hz')

plt.figure(figsize=(7,3))
librosa.display.specshow(f4, sr=22050, x_axis='time', y_axis='hz')

plt.figure(figsize=(7,3))
librosa.display.specshow(f5, sr=22050, x_axis='time', y_axis='hz')

# %%

""" Test for voice """

data, sr = librosa.core.load('./testrecord.wav')

# %%

plt.figure(figsize=(14,6))
librosa.display.waveplot(data, sr=sr)

# %%

X = librosa.stft(data)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

# %%

spect = librosa.feature.melspectrogram(y=data, sr=sr,n_fft=2048, hop_length=512)
spect = librosa.power_to_db(spect, ref=np.max)
plt.figure(figsize=(14,5))
librosa.display.specshow(spect, sr=22050, x_axis='time', y_axis='hz')

# %%

f2.shape

# %%

mfccs = librosa.feature.mfcc(data, sr=sr, n_mfcc=40, n_fft=2049, hop_length=512)
plt.figure(figsize=(14,5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# %%

mfccs = DataPreprocessor.get_mfcc('./testrecord.wav')
plt.figure(figsize=(14,5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# %%

""" Model Class """

class TrainingModels:
    
    def buildModel1(shape=(None, 128, 1320)):
        in_tensor = Input(shape=(None, None))
        tensor = Conv1D(512, kernel_size=(3,), activation='relu') (in_tensor)
        tensor = MaxPool1D(264)(tensor)
        tensor = Conv1D(256, kernel_size=(3,), activation='relu') (tensor)
        tensor = MaxPool1D(128)(tensor)
        tensor = Conv1D(64, kernel_size=(3,), activation='relu') (tensor)
        tensor = MaxPool1D(32)(tensor)

        tensor = Flatten() (tensor)

        tensor = Dense(16, activation='relu')(tensor)
        tensor = Dense(1, activation='sigmoid') (tensor)

        model = Model (in_tensor, tensor)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

# %%

# Train model

def buildModel1(shape=(128, 1320)):
    in_tensor = Input(shape=shape)
    tensor = Conv1D(512, kernel_size=(3,), activation='relu') (in_tensor)
    tensor = MaxPooling1D(2)(tensor)
    

    tensor = Flatten() (tensor)

    tensor = Dense(16, activation='relu')(tensor)
    tensor = Dense(11, activation='sigmoid') (tensor)

    model = Model (in_tensor, tensor)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# %%

model = buildModel1()
model.summary()

# %%

model.fit(
    x=dataGen.data,
    y=dataGen.labels,
    batch_size=4,
    verbose=1,
    epochs=100
)


# %%

def build_model_2D(shape):
    in_tensor = Input(shape=shape)

    # 2D Convolution Layer 1
    tensor = Conv2D(256, kernel_size=(3,3), data_format="channels_last", activation='relu', padding='valid')(in_tensor)
    tensor = MaxPooling2D(2)(tensor)

    # 2D Convolution Layer 2
    tensor = Conv2D(128, kernel_size=(3,3), data_format="channels_last", activation='relu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    # 2D Convolution Layer 3
    tensor = Conv2D(64, kernel_size=(3,3), data_format="channels_last", activation='relu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    # Flatten
    tensor = Flatten()(tensor)

    # Dense
    tensor = Dense(32, activation='relu')(tensor)
    tensor = Dense(16, activation='relu')(tensor)
    tensor = Dense(11, activation='sigmoid')(tensor)

    # Compile
    model = Model(in_tensor, tensor)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# %%


def build_model_2D_2(shape):
    in_tensor = Input(shape=shape)

    # 2D Convolution Layer 1
    tensor = Conv2D(128, kernel_size=(3,3), data_format="channels_last", activation='elu', padding='valid')(in_tensor)
    tensor = MaxPooling2D(2)(tensor)

    # 2D Convolution Layer 2
    tensor = Conv2D(64, kernel_size=(3,3), data_format="channels_last", activation='elu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    # 2D Convolution Layer 3
    tensor = Conv2D(32, kernel_size=(3,3), data_format="channels_last", activation='elu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    # 2D Convolution Layer 4
    tensor = Conv2D(16, kernel_size=(3,3), data_format="channels_last", activation='elu', padding='valid')(tensor)
    tensor = MaxPooling2D(2)(tensor)

    # Flatten
    tensor = Flatten()(tensor)
    
    # Dense
    tensor = Dense(512, activation='elu')(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(16, activation='elu')(tensor)
    tensor = Dense(11, activation='softmax')(tensor)

    # Compile
    model = Model(in_tensor, tensor)
    rmsOp = RMSprop(lr=0.0001)
    model.compile(optimizer=rmsOp, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# %%

model = build_model_2D_2(shape=dataGen.data.shape[1:])
model.summary()

# %%

dataGen.data.shape

# %%

# Normalized data
spect = DataPreprocessor.get_spect(dataGen.fname[0])
plt.figure(figsize=(14,5))
librosa.display.specshow(spect, sr=22050, x_axis='time', y_axis='hz')

# %%
data, sr = librosa.core.load(dataGen.fname[1])
spect = librosa.feature.melspectrogram(y=data, sr=sr,n_fft=2048, hop_length=512)
spect = librosa.power_to_db(spect, ref=np.max)
plt.figure(figsize=(14,5))
librosa.display.specshow(spect, sr=22050, x_axis='time', y_axis='hz')
# %%

model.save('model2D_3.h5')

# %%


# %%


model.predict(dataGen.data[1:2])
