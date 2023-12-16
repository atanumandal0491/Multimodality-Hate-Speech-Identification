import tensorflow as tf
from pe import PositionalEncoder



class Speech_Sampling(tf.keras.layers.Layer):
  def __init__(self, d_model, vocab_size, name="Speech_Sampling", **kwargs):
    super(Speech_Sampling, self).__init__(name=name, **kwargs)
    self.d_model = d_model
    self.vocab_size = vocab_size


    self.conv1 = tf.keras.layers.Conv1D(filters=4096, kernel_size=3, strides=1, padding='same')
    self.conv2 = tf.keras.layers.Conv1D(filters=1024, kernel_size=3, strides=2, padding='same')
    self.permute = tf.keras.layers.Permute((2, 1))

    self.lstm = tf.keras.layers.LSTM(units=d_model, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True, dropout=dropout_rate, return_sequences=True, stateful=False)

    self.pe = PositionalEncoder(vocab_size=vocab_size, d_model=d_model)

  def call(self, x):
    x = tf.nn.gelu(self.conv1(x))
    x = tf.nn.gelu(self.conv2(x))
    x = self.permute(x)
    x = self.lstm(x) + self.pe(x)
    return x
    
class Text_Sampling(tf.keras.layers.Layer):
  def __init__(self, d_model, vocab_size, name="Text_Sampling", **kwargs):
    super(Text_Sampling, self).__init__(name=name, **kwargs)
    self.d_model = d_model
    self.vocab_size = vocab_size

    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pe = PositionalEncoder(vocab_size=vocab_size, d_model=d_model)


  def call(self, x):
    x =  self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pe(x)
    return x
