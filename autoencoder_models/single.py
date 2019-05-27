from tensorflow.keras.layers import Input, Dense #prefix this with tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model

def run(input_dim, encoding_dim):

    #Input() is used to instantiate a Keras tensor

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)

    decoder = Dense(input_dim, activation='relu')(encoder)

    #you either use a keras model or build one yourself as we do here
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder