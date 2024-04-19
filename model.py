import tensorflow as tf

# Diffusion Model

def loss_func():
    pass

def accuracy_func():
    pass

class DiffusionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        # Where our main layers will be initialized
        # There should be an encoder and a decoder...

        self.prior = tf.keras.layers.Identity()


        super().__init__(**kwargs)
        pass
        
    def call():
        # Where we actually call the layers
        # Q: Should we be inputting the CLIP embedding directly?
        pass

