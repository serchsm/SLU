import tensorflow as tf


class TranformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, rate=0.1):
        super(TranformerEncoder, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop_1 = tf.keras.layers.Dropout(rate=rate)
        self.drop_2 = tf.keras.layers.Dropout(rate=rate)
