import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
import clip_wrapper

# 64x64 Model for Reference
# Please check the Final Notebooks for the most up-to-date, accurate model
class ClipCVAE(tf.keras.Model):

  def __init__(self, input_shape,  latent_dim, dropout_rate):
    super(ClipCVAE, self).__init__()
    self.latent_dim = latent_dim
    self.shape_input = input_shape
    self.dropout_prob = dropout_rate

    #Dense layer used to shrink the embedding before inputing it in the decoder
    self.embedding_shrinker = tf.keras.layers.Dense(32)

    #First part of the encoder before embedding concatenation
    self.encoder_part1= tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=self.shape_input, name="encoder_part1_inputlayer"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(256,5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d3"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d4"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten()])

    #Second part of the encoder after concatenation
    self.encoder_part2 = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, kernel_initializer=tf.keras.initializers.HeNormal(), activation = "leaky_relu"),
            tf.keras.layers.Dense(2048, kernel_initializer=tf.keras.initializers.HeNormal(), activation="leaky_relu"),
            tf.keras.layers.Dense(1024, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()

        ]
    )

    self.mu = tf.keras.layers.Dense(self.latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())
    self.logv = tf.keras.layers.Dense(self.latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())


    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8 * 8 * 256, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((8,8,256)),
            tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32, 5, strides=1, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(3, 5, strides=1, padding="same", activation="sigmoid", kernel_initializer=tf.keras.initializers.HeNormal())
        ]
    )

  # Loss combining Binary Cross entropy, KL Divergence and Structural Similarity
  def compute_loss(self, inputs, reconstructed, logv, mu):
    inputs, _ = inputs
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, reconstructed)))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(inputs, reconstructed, max_val=1.0))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logv - tf.square(mu) - tf.exp(logv)))
    total_loss = reconstruction_loss + 0.5 * kl_loss + 0.1 * ssim_loss
    return total_loss / BATCH_SIZE


  def call(self, args, training=False):
    x, embedding = args
    embedding = tf.squeeze(embedding, axis=1)

    # First part of the encoding
    e1 = self.encoder_part1(x)
    e1_w_embedding = tf.concat([e1, embedding], axis=-1)

    # Go through part 2 of encoder
    e2 = self.encoder_part2(e1_w_embedding)

    # Mu and Logv
    mu = self.mu(e2)
    logv = self.logv(e2)

    # Sample Z (reparameterization trick)
    sigma = tf.math.sqrt(tf.math.exp(logv))
    eps = tf.random.normal([BATCH_SIZE, self.latent_dim])
    z = mu + tf.multiply(sigma, eps)

    if training:
      dropout_rate = self.dropout_prob
      z = tf.nn.dropout(z, rate=dropout_rate)

    embedding = self.embedding_shrinker(embedding)
    # Concatenate z with embedding again
    z_w_embedding = tf.concat([z, embedding], axis=-1)

    # Decode!
    res = self.decoder(z_w_embedding)

    return res, mu, logv

  # Called when .fit is called
  def train_step(self, data):
      with tf.GradientTape() as tape:
          reconstruction, mu, logv = self(data, training=True)
          loss = self.compute_loss(data, reconstruction, logv, mu)
      grads = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      return {'train loss': loss}

  # Called at each epoch when .fit is called
  def test_step(self, data):
    reconstruction, mu, logv = self(data, training=False)
    loss = self.compute_loss(data, reconstruction, logv, mu)
    return {'valid loss': loss}

  #Visualisation
  def show_image(self, capt):
    z = tf.random.normal(shape=[1, self.latent_dim])
    encoding = clip_wrapper.get_text_encoding(capt)
    encoding = self.embedding_shrinker(encoding)
    z = tf.concat([z, encoding], axis=-1)
    img = self.decoder(z).numpy()[0]
    plt.imshow(img)