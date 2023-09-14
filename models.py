import tensorflow as tf
import keras

class MLP(keras.Model):
    """Multi-layer Perceptron"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.hidden_1 = keras.layers.Dense(256, activation="relu")
        self.hidden_2 = keras.layers.Dense(128, activation="relu")
        self.logit = keras.layers.Dense(num_classes)

    def call(self, x):
        y = self.flatten(x)
        y = self.hidden_1(y)
        y = self.hidden_2(y)
        return self.logit(y)

class ConvNet(keras.Modell):
    """Simple 2D ConvNet"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(8, (3, 3), input_shape=input_shape)
        self.maxpool = keras.layers.MaxPooling2D((2, 2))
        self.flatten = keras.layers.Flatten()
        self.logit = keras.layers.Dense(num_classes)
    
    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.maxpool(y)
        y = self.flaten(y)
        return self.logit(y)
    
class LeNet5(keras.Model):
    """LeNet5 architecture"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu", input_shape=input_shape)
        self.maxpool = keras.layers.MaxPool2D(2, 2)

        self.conv_2 = keras.layers.Conv2D(48, kernel_size=(5, 5), padding="valid", activation="relu")
        

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.dense_2 = keras.layers.Dense(84, activation="relu")
        self.logit = keras.layers.Dense(num_classes)
        
    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.maxpool(y)
        y = self.conv_2(y)
        y = self.maxpool(y)
        y = self.flatten(y)
        y = self.dense_1(y)
        y = self.dense_2(y)
        return self.logit(y)

class Sampling(keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_conv_encoder(input_shape=(28, 28, 1), latent_dim=2):
    latent_dim = latent_dim

    encoder_inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", strides=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation="relu")(x)

    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def create_conv_decoder(latent_dim=2):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation="relu", strides=2, padding="same")(x)
    x = keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation="relu", strides=2, padding="same")(x)
    decoder_outputs = keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def call(self, inputs, training=False):
        z_mean, _, z = self.encoder(inputs)
        if training:
            x = self.decoder(z)
        else:
            x = self.decoder(z_mean)

        return x
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    
    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         z_mean, z_log_var, z = encoder(data)
    #         reconstruction = decoder(z)
    #         reconstruction_loss = tf.reduce_mean(
    #             tf.reduce_sum(
    #                 keras.losses.binary_crossentropy(data, reconstruction), axis=(1 , 2)
    #             )
    #         )
    #         kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #         kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #         total_loss = reconstruction_loss + kl_loss
        
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
    #     self.total_loss_trakcer.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     self.kl_loss_tracker.update_state(kl_loss)
        
    #     return {
    #         "loss": self.total_loss_trakcer.result(),
    #         "reconstruction_loss": self.reconstruction_loss_tracker.result(),
    #         "kl_loss": self.kl_loss_tracker.result()
    #     }