import datetime
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


def initVariables():
    #- Exploring the data
    df = pd.read_csv("data/creditcard.csv")
    print(df)
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
    #print('ytest')
    #print(y_test)
    y_test .to_pickle('y_test.pkl')
    X_test = X_test.drop(['Class'], axis=1)

    X_train.to_pickle('x_train.pkl')
    X_test.to_pickle('x_test.pkl')

#initVariables() TODO make conditional
#X_train.shape
X_train = pd.read_pickle('x_train.pkl')
X_test = pd.read_pickle('x_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

X_train = X_train.values
X_test = X_test.values

#- Building model
input_dim = X_train.shape[1]
print("input dimension")
print(input_dim)
print("x train shape")
print(X_train.shape)
print("x test shape")
print(X_test.shape)

nb_epoch = 5
batch_size = 32
encoding_dim = 14
type='default'
optimizer = 'adam'
loss = 'mean_squared_error'
regularizerInput= 10e-5
activity_regularizer=regularizers.l1(regularizerInput)
title = str(datetime.datetime.now().strftime('%Y-%m-%d')) + '-ae-' + type + '-loss:' + loss + '-optimizer:' + optimizer + '-encoding_dim:' + str(encoding_dim) + '-epoch:' + str(nb_epoch) + '-batch-size:'+ str(batch_size) + '-regularizers-l1:' + str(regularizerInput)
title = title.replace('.', 'dot')

if type == 'default':
    from autoencoder_models.default import run
elif type == 'sixlayer':
    from autoencoder_models.sixlayer import run
elif type == 'fourlayer':
    from autoencoder_models.fourlayer import run
elif type == 'single':
    from autoencoder_models.single import run
else:
    print('unknown algorithm')
autoencoder = run(input_dim, encoding_dim, activity_regularizer)


#Once your model looks good, configure its learning process with .compile()
autoencoder.compile(optimizer=optimizer, 
                    loss=loss, 
                    metrics=['accuracy', 'binary_crossentropy'])

checkpointer = ModelCheckpoint(filepath="models/"+title+"-model.h5",
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
pathPrefix = 'results/'

#Model evaluation
print('history')
print(history)

fig = plt.figure(title + '__loss')
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()
plt.savefig(pathPrefix + title + '__loss')

print('history')
print(history)

# summarize history for accuracy
fig = plt.figure(title + '__accuracy')
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.draw()
plt.savefig(pathPrefix + title + '__accuracy')

scores = autoencoder.evaluate(X_train, X_train)
#print('ok')
#print("\n%s: %.2f%%" % (autoencoder.metrics_names[1], scores[1]*100))
#print('/ok')

#make predictions on new data/test data
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
print(error_df.describe())



from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
threshold_fixed = 20
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, pred_y)
print(conf_matrix)


predictions = autoencoder.predict(X_test)[:, 1]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(error_df.true_class, predictions)

print('fpr_keras')
print(fpr_keras)
print('tpr_keras')
tpr_keras
print('thresholds')
print(thresholds_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_keras, tpr_keras, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (0, auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.draw()
plt.savefig(pathPrefix + title + '__roc')
# Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()


plt.figure(title + '__confusion matrix')
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.draw()
plt.savefig(pathPrefix + title + '__confusion matrix')

#show all plots
#plt.show()
