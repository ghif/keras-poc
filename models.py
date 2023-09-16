import tensorflow as tf
import keras

def make_mlp(input_shape=(28, 28, 1), num_classes=10):
    """
    Multi-layer Perceptraon -- functional API

    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

class MLPSeq(keras.Sequential):
    """
    Multi-layer Perceptraon -- sequential model

    """
    def __init__(self, num_classes):
        super().__init__()
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(256, activation="relu"))
        self.add(keras.layers.Dense(128, activation="relu"))
        self.add(keras.layers.Dense(num_classes))
        

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
    
class MLPBN(keras.Model):
    """Multi-layer Perceptron"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.hidden_1 = keras.layers.Dense(512, activation="relu")
        self.bn_1 = keras.layers.BatchNormalization()

        self.hidden_2 = keras.layers.Dense(256, activation="relu")
        self.bn_2 = keras.layers.LayerNormalization()

        self.hidden_3 = keras.layers.Dense(128, activation="relu")
        self.logit = keras.layers.Dense(num_classes)

    def call(self, x):
        y = self.flatten(x)
        y = self.hidden_1(y)
        y = self.bn_1(y)

        y = self.hidden_2(y)
        y = self.bn_2(y)

        y = self.hidden_3(y)

        return self.logit(y)


class ConvNetSeq(keras.Sequential):
    """Simple 2D ConvNet"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.add(keras.layers.Conv2D(8, (3, 3), input_shape=input_shape))
        self.add(keras.layers.MaxPooling2D((2, 2)))
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(num_classes))
    

class ConvNet(keras.Model):
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
        y = self.flatten(y)
        return self.logit(y)

class GhifNet5(keras.Model):
    """LeNet5 architecture"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", input_shape=input_shape)
        self.bn_1 = keras.layers.BatchNormalization()
        self.maxpool_1 = keras.layers.MaxPool2D(2, 2)

        self.conv_2 = keras.layers.Conv2D(48, kernel_size=(5, 5), padding="same")
        self.bn_2 = keras.layers.BatchNormalization()
        
        self.maxpool_2 = keras.layers.MaxPool2D(2, 2)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.dense_2 = keras.layers.Dense(128, activation="relu")

        self.logit = keras.layers.Dense(num_classes)
        
    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn_1(y)
        y = self.maxpool_1(y)

        y = self.conv_2(y)
        y = self.bn_2(y)
        y = self.maxpool_2(y)
        y = self.flatten(y)

        y = self.dense_1(y)
        y = self.dense_2(y)

        return self.logit(y)
    
class LeNet5Seq(keras.Sequential):
    """
    LeNet5 architecture -- Sequential
    """
    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.add(keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu", input_shape=input_shape))
        self.add(keras.layers.MaxPool2D(2, 2))
        self.add(keras.layers.Conv2D(48, kernel_size=(5, 5), padding="valid", activation="relu"))
        self.add(keras.layers.MaxPool2D(2, 2))

        self.add(keras.layers.Flatten())

        self.add(keras.layers.Dense(256, activation="relu"))
        self.add(keras.layers.Dense(84, activation="relu"))
        self.add(keras.layers.Dense(num_classes))
        
class LeNet5(keras.Model):
    """LeNet5 architecture"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        super().__init__()
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu", input_shape=input_shape)
        self.maxpool_1 = keras.layers.MaxPool2D(2, 2)

        self.conv_2 = keras.layers.Conv2D(48, kernel_size=(5, 5), padding="valid", activation="relu")
        
        self.maxpool_2 = keras.layers.MaxPool2D(2, 2)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(256, activation="relu")
        self.dense_2 = keras.layers.Dense(84, activation="relu")
        self.logit = keras.layers.Dense(num_classes)
        
    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.maxpool_1(y)
        y = self.conv_2(y)
        y = self.maxpool_2(y)
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
        z_mean, z_log_var, z = self.encoder(inputs)
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
    
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.compute_loss(y=x, y_pred=reconstruction)
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }