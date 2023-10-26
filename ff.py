import tensorflow as tf

class Sequential_Sampling(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate, name="Sequential_Sampling", **kwargs):
    super(Sequential_Sampling, self).__init__(name=name, **kwargs)
    self.d_model = d_model
    self.dff = dff
    self.dropout_rate = dropout_rate

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)


  def call(self, x):
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dropout(x)
    return x



class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.dff = dff
    self.dropout_rate = dropout_rate

    self.seq = Sequential_Sampling(d_model, dff, dropout_rate)
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):

    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

