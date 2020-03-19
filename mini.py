# %%
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers import Input, Dense, Embedding, Flatten, Dropout, merge, Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import add, concatenate
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from sklearn import preprocessing
from keras.utils import plot_model
import keras
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
#training_data=pd.read_csv('C:/Users/ITYENDU/Desktop/New Myopinions dataset/New folder/rating_table.csv')
training_data=pd.read_csv('rating_table.csv',header=None)
training_data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
training_data.head()

# %%
#training_data.loc[training_data.UserID == 132]
training_data.shape

# %%
training_data =training_data .drop(['Timestamp' ], axis='columns')
training_data

# %%
training_user_movie_matrix=training_data.pivot_table(index='UserID',columns='MovieID',values='Rating',fill_value=0)
training_user_movie_matrix.head()

# %%
testing_data=pd.read_csv('testing_data.csv',header=None)
testing_data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
testing_data.head()

# %%
testing_user_movie_matrix=training_data.pivot_table(index='UserID',columns='MovieID',values='Rating',fill_value=0)
testing_user_movie_matrix.head()

# %%
#max_value = 5
#training_user_movie_matrix =training_user_movie_matrix.astype('float32') / max_value
#testing_user_movie_matrix = testing_user_movie_matrix.astype('float32') / max_value
print(training_user_movie_matrix.shape)
print(testing_user_movie_matrix.shape)

# %%
input_data= Input(shape=(728,))
encoded1 = Dense(units=256, activation='relu')(input_data)
encoded2 = Dense(units=128, activation='relu')(encoded1)
encoded3 = Dense(units=64, activation='relu')(encoded2)
encoded4 = Dense(units=32, activation='relu',name = "encoder_out")(encoded3)
decoded4 = Dense(units=64, activation='relu')(encoded4)
decoded3 = Dense(units=128, activation='relu')(decoded4)
decoded2 = Dense(units=256, activation='relu')(decoded3)
decoded1 = Dense(units=728, activation='sigmoid')(decoded2)

autoencoder=Model(input_data, decoded1)
autoencoder.summary()


# %%
autoencoder.compile(optimizer='adam', loss='mse')
model=autoencoder.fit(training_user_movie_matrix,training_user_movie_matrix, epochs=500, batch_size=256, shuffle=True,  
                validation_data=(testing_user_movie_matrix,testing_user_movie_matrix))


def king_plot(hist):
    plt.title('Model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss') 
    plt.legend(['train','test'],loc='upper left')
    plt.subplot(3,1,1)
    plt.plot(hist.history['loss'])
    plt.subplot(3,1,2)
    plt.plot(hist.history['val_loss'])
    plt.subplot(3,1,3)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.show()
    plt.savefig('three.png')

king_plot(model)

# %%
encoder_out = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer("encoder_out").output)
inter_out = encoder_out.predict(training_user_movie_matrix).round(2)
inter_out

# %%
print(inter_out.shape)

# %%
np.savetxt("bottleneck out predictions.csv",inter_out, delimiter=",")


# %%
user_movie_predictions=pd.read_csv("bottleneck out predictions.csv",header=None)
user_movie_predictions=user_movie_predictions.round(2)
user_movie_predictions.index=testing_user_movie_matrix.index

# %%
user_movie_predictions.head(5)

# %%
user_count = (testing_data.(
     groupby(by = ['UserID'])['Rating'].)
     count())

user_count = pd.DataFrame(user_count)
user_count.columns = ['Rating_count']
user_count.head(10)

# %%
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(user_movie_predictions)#testing_data.drop(['Timestamp'],axis='columns'))
#_user_movie_matrix)

# %%
list4 = []
list3 = []
for i in range(0,634):
    query_index = np.random.choice(user_movie_predictions.shape[0])
    distances, indices = model_knn.kneighbors(user_movie_predictions.iloc[query_index, :].values.reshape(1,-1), n_neighbors = 6)
    list1 = []
    list2 = []
    for j in range(0, len(distances.flatten())):
        if j == 0:
            x=5
            #print('Recommendations for {0}:\n'.format(user_movie_predictions.index[query_index]))
        else:
            list1.append(user_movie_predictions.index[indices.flatten()[j]])
            list2.append(distances.flatten().round(5)[j])
            #print('{0}: {1}, with distance of {2}:'.format(j, user_movie_predictions.index[indices.flatten()[j]], distances.flatten()[j]))
    list4.append(list1)
    list3.append(list2)
'''
print("Movie Id: \n")
print(list4)
print("Nearest neighbor distance: \n")
print(list3)

print("\n")
print(len(list4))
print("\n")
print(len(list3))
'''
X = pd.DataFrame(list4)
X.to_csv('one.csv',index=False)

Y = pd.DataFrame(list3)
Y.to_csv('two.csv',index=False)
