import pandas as pd
import numpy as np
import math

df = pd.read_csv('temperature.csv',delimiter=";",decimal=",")
data = pd.read_csv('temperature2.csv')
df = df[df['DateTime'].notna()]
df = df.reset_index(drop=True)

df= df.drop('DateTime', axis = 1)


hours=[12.25-(abs(12.25-((i)%24)))/12.25 for i in range(26304)]

df['Hour']=hours

seasons=[((i+720)//(2160))%4 for i in range(26304)]
#df= df.drop('WWCode', axis = 1)
df['WWCode'] = df['WWCode'].fillna(-1)

df['Season']=seasons
df['Month']=[(i//730)%12 for i in range(26304)]
df['Zenith'] = data['Zenith']
df['cloud'] = data['Cloudopacity']
df['GHI'] = data['GHI']


df['Day']=[(i//24) for i in range(26304)]
df['DayLength']=[9+(6* math.sin(math.radians(((i+264)%8760)/48.6666))) for i in range(26304)]

df['CloudPrev'] = [0 if i == 0 else data['Cloudopacity'][i-1] for i in range(26304)]
df['DewPointprev'] = [0 if i == 0 else data['DewPoint'][i-1] for i in range(26304)]
df['winddir2'] = [math.cos(math.radians(df['WindDirection'][i])) for i in range(26304)]
df['winddir3'] = [math.cos(math.radians(df['WindDirection'][i])) for i in range(26304)]




nd = pd.read_csv('generation.csv',delimiter=';',decimal=',')

nd = nd[nd['DateTime'].notna()]
nd = nd.reset_index(drop=True)
nd = nd.drop('DateTime',axis=1)



features = df[:25560]
features = np.array(features)

features = features.reshape((features.shape[0], features.shape[1], 1))

outcomes = np.array(nd)


from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(features.shape[1], features.shape[2])))
model.add(Dense(8, activation='relu'))

model.add(Dense(1,activation='linear'))

#opt = keras.optimizers.Adam(learning_rate=0.002)

model.compile(loss='mse', optimizer='adam')

model.summary()


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.2f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_weights_only=True, save_best_only = True, mode ='min')
callbacks_list = [checkpoint]

model.fit(features,outcomes,validation_split=0.2, epochs=100, batch_size=32,callbacks=callbacks_list)

print("TRAINING DONE!")
