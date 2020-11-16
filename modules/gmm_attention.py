from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, AttentionMechanism
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# def _GmmAttention_score(processed_query, mu, delta, w,
#                     keys,
#                     attention_v,
#                     Z=1,
#                     attention_g=None,
#                     attention_b=None):
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
#   processed_query = array_ops.expand_dims(processed_query, 1)
#   score = reduce_sum((w/k)*)
#   if attention_g is not None and attention_b is not None:
#     normed_v = attention_g * attention_v * math_ops.rsqrt(
#         math_ops.reduce_sum(math_ops.square(attention_v)))
#     return math_ops.reduce_sum(
#         normed_v * math_ops.tanh(keys + processed_query + attention_b), [2])
#   else:
#     return math_ops.reduce_sum(
#         attention_v * math_ops.tanh(keys + processed_query), [2])


# class GmmAttention(_BaseAttentionMechanism):

#   def __init__(self,
#                num_units,
#                memory,
#                delta=0,
#                mu=0,
#                memory_sequence_length=None,
#                normalize=False,
#                probability_fn=None,
#                score_mask_value=None,
#                dtype=None,
#                custom_key_value_fn=None,
#                name="GmmAttention"):
#     """Construct the Attention mechanism.

#     Args:
#       num_units: The depth of the query mechanism.
#       mu: The initial mu valuef of the GMM model. (batch_size, )
#       memory: The memory to query; usually the output of an RNN encoder.  This
#         tensor should be shaped `[batch_size, max_time, ...]`.
#       memory_sequence_length: (optional) Sequence lengths for the batch entries
#         in memory.  If provided, the memory tensor rows are masked with zeros
#         for values past the respective sequence lengths.
#       normalize: Python boolean.  Whether to normalize the energy term.
#       probability_fn: (optional) A `callable`.  Converts the score to
#         probabilities.  The default is `tf.nn.softmax`. Other options include
#         `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
#         Its signature should be: `probabilities = probability_fn(score)`.
#       score_mask_value: (optional): The mask value for score before passing into
#         `probability_fn`. The default is -inf. Only used if
#         `memory_sequence_length` is not None.
#       dtype: The data type for the query and memory layers of the attention
#         mechanism.
#       custom_key_value_fn: (optional): The custom function for
#         computing keys and values.
#       name: Name to use when creating ops.
#     """
#     self.mu = mu
#     if delta == 0:
#         self.delta = 1
#     if probability_fn is None:
#       probability_fn = nn_ops.softmax
#     if dtype is None:
#       dtype = dtypes.float32
#     wrapped_probability_fn = lambda score, _: probability_fn(score)
#     super(GmmAttention, self).__init__(
#         query_layer=layers_core.Dense(
#             num_units, name="query_layer", use_bias=False, dtype=dtype),
#         memory_layer=layers_core.Dense(
#             num_units, name="memory_layer", use_bias=False, dtype=dtype),
#         memory=memory,
#         probability_fn=wrapped_probability_fn,
#         custom_key_value_fn=custom_key_value_fn,
#         memory_sequence_length=memory_sequence_length,
#         score_mask_value=score_mask_value,
#         name=name)
#     self._num_units = num_units
#     self._normalize = normalize
#     self._name = name

#   def __call__(self, query, state):
#     """Score the query based on the keys and values.

#     Args:
#       query: Tensor of dtype matching `self.values` and shape `[batch_size,
#         query_depth]`.
#       state: Tensor of dtype matching `self.values` and shape `[batch_size,
#         alignments_size]` (`alignments_size` is memory's `max_time`).

#     Returns:
#       alignments: Tensor of dtype matching `self.values` and shape
#         `[batch_size, alignments_size]` (`alignments_size` is memory's
#         `max_time`).
#     """
#     with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
#       processed_query = self.query_layer(query) if self.query_layer else query
#       attention_v = variable_scope.get_variable(
#           "attention_v", [self._num_units], dtype=query.dtype)
#       if not self._normalize:
#         attention_g = None
#         attention_b = None
#       else:
#         attention_g = variable_scope.get_variable(
#             "attention_g",
#             dtype=query.dtype,
#             initializer=init_ops.constant_initializer(
#                 math.sqrt((1. / self._num_units))),
#             shape=())
#         attention_b = variable_scope.get_variable(
#             "attention_b", [self._num_units],
#             dtype=query.dtype,
#             initializer=init_ops.zeros_initializer())

#       score = _GmmAttention_score(
#           processed_query,
#           self.mu,
#           self.delta,
#           self._keys,
#           attention_v,
#           attention_g=attention_g,
#           attention_b=attention_b)
#     alignments = score#self._probability_fn(score, state)
#     next_state = alignments
#     return alignments, next_state

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from functools import partial

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class GmmAttention(attention_wrapper.AttentionMechanism):
    def __init__(self,
                 memory,
                 num_mixtures=16,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):

        self.dtype = memory.dtype
        self.num_mixtures = num_mixtures
        self.query_layer = tf.layers.Dense(
            3 * num_mixtures, name='gmm_query_layer', use_bias=True, dtype=self.dtype)

        with tf.name_scope(name, 'GmmAttentionMechanismInit'):
            if score_mask_value is None:
                score_mask_value = 0.
            self._maybe_mask_score = partial(
                attention_wrapper._maybe_mask_score,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value)
            self._value = attention_wrapper._prepare_memory(
                memory, memory_sequence_length, check_inner_dims_defined)
            self._batch_size = (
                self._value.shape[0].value or tf.shape(self._value)[0])
            self._alignments_size = (
                    self._value.shape[1].value or tf.shape(self._value)[1])

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        state_size_ = self.state_size
        return _zero_state_tensors(state_size_, batch_size, dtype)

    def __call__(self, query, state):
        with tf.variable_scope("GmmAttention"):
            previous_kappa = state
            
            params = self.query_layer(query)
            print(params.shape, state.shape)
            alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

            # [batch_size, num_mixtures, 1]
            alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)
            # softmax makes the alpha value more stable.
            # alpha = tf.expand_dims(tf.nn.softmax(alpha_hat, axis=1), axis=2)
            beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
            kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

            # [1, 1, max_input_steps]
            mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32),
                            shape=[1, 1, self.alignments_size])

            # [batch_size, max_input_steps]
            phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 2.), axis=1)

        alignments = self._maybe_mask_score(phi)
        next_state = tf.squeeze(kappa, axis=2)

        return alignments, next_state