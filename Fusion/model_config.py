import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from pre import Speech_Sampling, Text_Sampling

class WhispeClassifierGenerator(tf.keras.models.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_classes, dropout_rate, **kwargs):
    super(WhisperClassifierGenerator, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

    self.speech_pre = Speech_Sampling(d_model=d_model, vocab_size=input_vocab_size)
    self.text_pre = Text_Sampling(d_model=d_model, vocab_size=target_vocab_size)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)

  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, inputs):
    inp1, inp2  =  inputs

    inp1 = self.speech_pre(inp1)
    inp1 = self.encoder(inp1)

    inp2 = self.text_pre(inp2)
    out = self.decoder(inp2, inp1)

    return out

'''
class WhisperClassifierDiscriminator(tf.keras.models.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_classes, dropout_rate, **kwargs):
    super(WhisperClassifierDiscriminator, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

    self.speech_pre = Speech_Sampling(d_model=d_model, vocab_size=input_vocab_size)
    self.text_pre = Text_Sampling(d_model=d_model, vocab_size=target_vocab_size)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)


  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, inputs):
    inp1, inp2  =  inputs  #inp1 --> Speech; inp2 --> Text

    inp1 = self.speech_pre(inp1)
    inp2 = self.text_pre(inp2)

    inp2 = self.encoder(inp2)
    out = self.decoder(inp1, inp2)


    try:
      del out._keras_mask
    except AttributeError:
      pass


    return out

'''

class WhisperClassifier(tf.keras.models.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_classes, dropout_rate, **kwargs):
    super(WhisperClassifier, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

    self.generator = WhisperClassifierGenerator(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, num_classes=num_classes, dropout_rate=dropout_rate)
    #self.discriminator = WhisperClassifierDiscriminator(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, num_classes=num_classes, dropout_rate=dropout_rate)

    self.flatten = tf.keras.layers.Flatten()


    #self.conc = tf.keras.layers.Concatenate(axis=1)
    #self.bi = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=d_model, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True, dropout=dropout_rate, return_sequences=False, stateful=False), merge_mode='sum')
    self.bi = tf.keras.layers.LSTM(units=d_model, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True, dropout=dropout_rate, return_sequences=False, stateful=False)

    #self.linear_layer_1 = tf.keras.layers.Dense(target_vocab_size, activation='relu')
    #self.linear_layer_2 = tf.keras.layers.Dense(int(d_model/8), activation='relu')
    self.final_layer = tf.keras.layers.Dense(num_classes)

  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, inputs):

    x1 = self.generator(inputs)

    #x2 = self.discriminator(inputs)
    #out = self.linear_layer_1(x1)
    #out = self.conc([x1, x2])

    out = self.bi(x1)

    #out = self.linear_layer_1(out)
    #out = self.flatten(out)

    #out = self.linear_layer_1(out)
    #out = self.linear_layer_2(out)
    out = self.final_layer(out)


    try:
      del out._keras_mask
    except AttributeError:
      pass


    return out

  def compute_loss(self, inputs, labels, training=True):
    if training:
      predictions = self(inputs, training=training)

      labels = tf.squeeze(tf.cast(labels, dtype=tf.int32), axis=-1)
      bincounts = tf.constant([7333, 2453])
      batch_weight = tf.cast(tf.gather(bincounts, labels), dtype=tf.float32)

      cal_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, predictions)
      loss = tf.math.reduce_sum(cal_loss*batch_weight)/(tf.math.reduce_sum(batch_weight) + tf.keras.backend.epsilon())
    else:
      predictions = self(inputs, training=training)
      labels = tf.squeeze(tf.cast(labels, dtype=tf.int32), axis=-1)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, predictions)
    return loss


