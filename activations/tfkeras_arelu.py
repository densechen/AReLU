import tensorflow as tf 

class ARelu(tf.keras.layers.Layer):
    def __init__(self, alpha=0.90, beta=2.0, **kwargs):
        super(ARelu, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta  = beta

    def call(self, inputs, training=None):
        alpha = tf.clip_by_value(self.alpha, clip_value_min=0.01, clip_value_max=0.99)
        beta  = 1 + tf.math.sigmoid(self.beta)
        return tf.nn.relu(inputs) * beta - tf.nn.relu(-inputs) * alpha
      
    def get_config(self):
        config = {
          'alpha': self.alpha,
          'beta': self.beta
        }
        base_config = super(ARelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
