from tensorflow.keras.layers import Input, Dense #prefix this with tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model

def run(input_dim):

    encoding_dim = 512

    #Input() is used to instantiate a Keras tensor
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)                      #256
    encoder = Dense(int(encoding_dim / 2 / 2), activation="relu")(encoder)                  #128
    encoder = Dense(int(encoding_dim / 2 / 2 / 2), activation="relu")(encoder)              #64
    encoder = Dense(int(encoding_dim / 2 / 2 / 2 / 2), activation="relu")(encoder)          #32
    encoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2), activation="relu")(encoder)     #16
    print(' layer')
    print(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2))
    encoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2), activation="relu")(encoder) #8
    print('last layer')
    print(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2 / 2))
    encoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2 / 2), activation="relu")(encoder) #4
    encoder = Dense(int(3), activation="relu")(encoder) #3

    decoder = Dense(int(3), activation='tanh')(encoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2 / 2), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2 / 2), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2 / 2), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 / 2 / 2), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 /2), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 / 2 ), activation='tanh')(decoder)
    decoder = Dense(int(encoding_dim  / 2 ), activation='tanh')(decoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    #you either use a keras model or build one yourself as we do here
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder