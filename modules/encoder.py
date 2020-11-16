from tensorflow._api.v1 import layers
from tensorflow.contrib.rnn.python.ops.rnn_cell import HighwayWrapper
from modules.layers import *
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops, array_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
import collections

class CBHGencoder():
    def __init__(self,
                 CBHG_num_layers=5,
                 CBHG_kernel_size=[1,2,3,4,5],
                 CBHG_channels=128,
                 CBHG_dropout_rate=0.0,
                 is_training=False,
                 activation_fn=tf.nn.tanh,
                 scope="CBHGencoder"):
        super(CBHGencoder, self).__init__()
        self.CBHG_num_layers = CBHG_num_layers
        self.CBHG_kernel_size = CBHG_kernel_size
        self.CBHG_channels = CBHG_channels
        self.CBHG_dropout_rate = CBHG_dropout_rate
        self.is_training = is_training
        self.activation_fn = activation_fn
        self.scope = scope
    
    def __call__(self, inputs, input_length = None):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.CBHG_num_layers - 1):
                #convbank layers
                x = conv1d_bn_drop(
                    inputs=inputs,
                    kernel_size=self.CBHG_kernel_size[i],
                    channels=self.CBHG_channels,
                    activation_fn=self.activation_fn,
                    is_training=self.is_training,
                    dropout_rate=self.CBHG_dropout_rate,
                    scope='Convbank_{}_'.format(i + 1) + self.scope,
                    )
            # x = conv1d_bn_drop(
            # inputs=x,
            # kernel_size=self.CBHG_kernel_size[-1],
            # channels=self.CBHG_channels,
            # activation_fn=self.activation_fn,
            # is_training=self.is_training,
            # dropout_rate=self.CBHG_dropout_rate,
            # scope='Convbank_{}_'.format(6) + self.scope,)
        
        # residual link
        residual_projection = CustomProjection(
                num_units=self.CBHG_channels,
                apply_activation_fn=False,
                scope='CBHG_res_projection')
        output = residual_projection(inputs) + x

        # Highway layers
        T = tf.nn.sigmoid(output, name='transform_gate')
        H = tf.nn.tanh(output, name = 'non_linear_transform')
        C = tf.subtract(1.0, T, name='carry_gate')
        outputs = tf.add(tf.multiply(H, T), tf.multiply(output, C), 'Highway_layer')
        # G = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(300,reset_after=True,
        #                     recurrent_activation='sigmoid',
        #                     return_state=True,
        #                     name = 'encoder_gru'))
        # outputs,_,_ = G(x)
        # outputs = tf.reshape(outputs, [-1, 1, 600])
        print(outputs.shape)
    #   print(highway.shape)

        return outputs


