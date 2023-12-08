import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from pre import Speech_Sampling, Text_Sampling

class ClassifierGenerator(tf.keras.models.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_classes, dropout_rate, **kwargs):
    super(ClassifierGenerator, self).__init__()

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



class Classifier(tf.keras.models.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, num_classes, dropout_rate, **kwargs):
    super(Classifier, self).__init__()

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff

    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

    self.generator = ClassifierGenerator(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, num_classes=num_classes, dropout_rate=dropout_rate)
    

    self.flatten = tf.keras.layers.Flatten()


  
    self.bi = tf.keras.layers.LSTM(units=d_model, activation='tanh', recurrent_activation='sigmoid', use_bias=True, unit_forget_bias=True, dropout=dropout_rate, return_sequences=False, stateful=False)

   
    self.final_layer = tf.keras.layers.Dense(num_classes)

  def get_config(self):
    config = super().get_config().copy()
    return config

  def call(self, inputs):

    x1 = self.generator(inputs)

    out = self.bi(x1)

   
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


