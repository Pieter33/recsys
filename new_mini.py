import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import warnings
import seaborn as sns

from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# read the data and make it ready for processing.
df = pd.read_csv('rating_table.csv')
df.columns = ['user_id','movie_id','rating','timestamp']
df = df.drop(['timestamp'], axis='columns')

df2 = pd.read_csv('testing_data.csv')
df2.columns = ['user_id','movie_id','rating','timestamp']
df2 = df2.drop(['timestamp'],axis='columns')

# df.head()
# df2.head()

# make (user_id x movie_id) matrix for both training and testing data.
df_matrix = df.pivot_table(index='user_id',columns='movie_id',values='rating').fillna(0)

df2_matrix = df2.pivot_table(index='user_id',columns='movie_id',values='rating').fillna(0)

# df_matrix.head()
# df2_matrix.head()

# make the model
# making the autoencoder model which encodes the original data and then reproduces it.
input_X = Input(shape=(728,))
encode1 = Dense(units=256,activation='relu')(input_X)
encode2 = Dense(units=128,activation='relu')(encode1)
encode3 = Dense(units=64,activation='relu')(encode2)
encode4 = Dense(units=32,activation='relu', name="encoder_out")(encode3)
encode4 = Dropout(0.3,name='Dropout')(encode4)
decode4 = Dense(units=64,activation='relu')(encode4)
decode3 = Dense(units=128,activation='relu')(decode4)
decode2 = Dense(units=256,activation='relu')(decode3)
decode1 = Dense(units=728,activation='sigmoid')(decode2)
# the neural network model is ready.

# using the model for our purpose.
autoencoder = Model(input_X, decode1)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')

model = autoencoder.fit(df_matrix.values, df_matrix.values, epochs=500, batch_size=256, shuffle=True, validation_data=(df2_matrix.values,df2_matrix.values))

def plot_hist(hist):
    fig, ax = plt.subplots()  # create figure & 1 axis

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.savefig("loss curve.png")

plot_hist(model)

# building the recommender
new_matrix = autoencoder.predict(df_matrix.values) * (df_matrix.values == 0)

new_matrix_df = pd.DataFrame(new_matrix, columns=df_matrix.columns, index = df_matrix.index)
new_matrix_df.round(3)
print(new_matrix_df.head()) # entire predicted_matrix

print(df_matrix.head())
np.savetxt("final matrix.csv", new_matrix_df)

#### a sample recommendation.
def recommend_me(user_id, matrix, topn = 6):
    pred_scores = matrix.loc[user_id].values
    df_score = pd.DataFrame({'movie_id':list(df_matrix.columns), 'score': pred_scores})
    
    df_record = df_score.set_index('movie_id').sort_values('score', ascending=False).head(topn)[['score']]
    

    return df_record[df_record.score > 0]

print(recommend_me(user_id=134, matrix=df_matrix))
print(recommend_me(user_id=134, matrix=new_matrix_df))

# ---------- latent space or bottlenecks
encoder = Model(inputs=autoencoder.input, outputs = autoencoder.get_layer("encoder_out").output)
inter_out = encoder.predict(df2_matrix).round(3)
#print(inter_out)

np.savetxt("bottle neck.csv", inter_out, delimiter=",")
# -------------- latent space matrix ------------------------

user_movie_predictions=pd.read_csv("bottle neck.csv", header=None)
user_movie_predictions=user_movie_predictions.round(3)
user_movie_predictions.index=df2_matrix.index

user_movie_predictions.head(5)


user_count = (df2.groupby(by = ['user_id'])['rating'].count())

user_count = pd.DataFrame(user_count)
user_count.columns = ['ratings_count']
# print user_count.head(10)

# %%
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(user_movie_predictions)

list4 = [] # movie_ids
list3 = [] # nearest neighbors distance

for i in range(0,634):
    query_index = np.random.choice(user_movie_predictions.shape[0])
    distances, indices = model_knn.kneighbors(user_movie_predictions.iloc[query_index, :].values.reshape(1,-1), n_neighbors = 6)
    list1 = []
    list2 = []
    for j in range(0, len(distances.flatten())):
        if j == 0:
            x=5
        else:
            list1.append(user_movie_predictions.index[indices.flatten()[j]])
            list2.append(distances.flatten().round(5)[j])

    list4.append(list1)
    list3.append(list2)

# nearest neighbors distance , save to csv
X = pd.DataFrame(list3)
X.to_csv('neighbors distance.csv', index=False)

# nearest movie_ids, save to csv
Y = pd.DataFrame(list4)
Y.to_csv('nearest movies.csv', index=False)
