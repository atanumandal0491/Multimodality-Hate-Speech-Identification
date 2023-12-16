import tensorflow as tf

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x, key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x, key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class Attentive_Fusion(tf.keras.layers.Layer):
  def __init__(self, num_dim, name="Hybrid Attention", **kwargs):
    super(Attentive_Fusion, self).__init__()
    self.num_dim = num_dim
    self.wq = tf.keras.layers.Dense(num_dim)
    self.wk = tf.keras.layers.Dense(num_dim)

  def call(self, x1, x2):
    q = self.wq(x1)
    k = self.wk(x2)
    qk = tf.linalg.matmul(q, k, transpose_a=False, transpose_b=True)
    weights = tf.math.exp(tf.math.tanh(qk))

    weights /= tf.cast(tf.math.reduce_sum(weights, axis=1, keepdims=True) + tf.keras.backend.epsilon(), dtype=tf.float32)
    weights = weights * qk
    weights = tf.math.reduce_sum(weights, axis=1)
    return weights
