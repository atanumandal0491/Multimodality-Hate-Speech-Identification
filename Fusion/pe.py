import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
  depth = depth/2
  positions = np.arange(length)[:, np.newaxis]
  depths = np.arange(depth)[np.newaxis, :]/depth
  angle_rates = 1 / (10000**depths)
  angle_rads = positions * angle_rates
  pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)


  def call(self, x):
    length = tf.shape(x)[1]
    x = self.pos_encoding[tf.newaxis, :length, :]
    return x

