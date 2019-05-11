import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense #prefix this with tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers

#plt optional settings
#sns.set(style='whitegrid', palette='muted', font_scale=1.5)
#rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

#- Exploring the data
df = pd.read_csv("data/creditcard.csv")
#5 first records of dataset
#print(df.head(n=5))
#print 'any null values:', df.isnull().values.any()

#- Preparing the data
from sklearn.preprocessing import StandardScaler

data = df.drop(['Time'], axis=1)

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
#only use non fraudulent for training set
X_train = X_train[X_train.Class == 0]
#drop class label for training set
X_train = X_train.drop(['Class'], axis=1)   

#list of all class labels only
y_test = X_test['Class']
print('ytest')
print(y_test)
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

X_train.shape


#- Building model
input_dim = X_train.shape[1]
print("input dimension")
print(input_dim)
print("x train shape")
print(X_train.shape)
print("x test shape")
print(X_test.shape)
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

#you either use a keras model or build one yourself as we do here
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 30
batch_size = 32

#Once your model looks good, configure its learning process with .compile()
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

#You can now iterate on your training data in batches:
#contains:
#loss
#val accuracy
#val loss
#accuracy
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

#A Keras model instance. 
#If an optimizer was found as part of the saved model, the model is already compiled. 
#Otherwise, the model is uncompiled and a warning will be displayed. 
#When compile is set to False, the compilation is omitted without any warning.
#see https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
autoencoder = load_model('model.h5')


#Model evaluation
print('history')
print(history)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.draw()


scores = autoencoder.evaluate(X_train, X_train)
#print('ok')
#print("\n%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))
#print('/ok')

#make predictions on new data
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
print(error_df.describe())



from sklearn.metrics import confusion_matrix, precision_recall_curve
threshold_fixed = 5
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, pred_y)
print(conf_matrix)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.draw()
plt.show()