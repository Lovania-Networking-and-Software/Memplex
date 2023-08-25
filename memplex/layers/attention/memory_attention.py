#  Copyright 2023 Lovania Networking and Software
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# =========================================================================
import numpy as np
import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

from memplex.layers.attention.attention import Attention


class MemoryAttention(Attention):

    @tf_utils.shape_type_conversion
    def build(
            self,
            input_shape
    ):
        self.beam_rift_space = self.add_weight(
            "beams",
            [input_shape[2], input_shape[0], input_shape[1], input_shape[2]],
            initializer=tf.keras.initializers.Zeros(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            trainable=False
        )
        beam_shape = self.compute_beam_shape(input_shape)
        self.m = self.add_weight(
            "memory",
            input_shape,
            initializer=tf.keras.initializers.Orthogonal(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            trainable=False
        )
        self.m_cache = self.add_weight(
            "memory_cache",
            input_shape,
            initializer=tf.keras.initializers.Zeros(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            trainable=True
        )

        kernel_size = conv_utils.normalize_tuple(
            beam_shape[1], 1, "kernel_size"
        )
        input_dim = int(beam_shape[-1])
        depthwise_kernel_shape = kernel_size + (
            input_dim,
            1
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=tf.keras.initializers.GlorotUniform(),
            name="depthwise_kernel",
            regularizer=None,
            constraint=None,
        )

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape))
        super().build(input_shape)

    def compute_beam_shape(self, input_shape):
        candidate_beams = self.beam_rift(tf.ones(input_shape), shape_computing=True)
        return candidate_beams.shape

    def beam_rift(self, x, shape_computing=False):
        beam_width = x.shape[2]

        probabilities = tf.nn.log_softmax(x)

        top_k_probabilities, top_k_indices = tf.nn.top_k(probabilities, k=beam_width)

        for bi in range(beam_width):
            old_top_k_probabilities = top_k_probabilities
            top_k_probabilities, top_k_indices = tf.nn.top_k(top_k_probabilities, k=1)
            if not shape_computing:
                if bi > 0:
                    self.beam_rift_space[bi].assign(
                        tf.nn.log_softmax(top_k_probabilities * self.beam_rift_space[bi - 1])
                    )
                else:
                    self.beam_rift_space[bi].assign(
                        tf.nn.log_softmax(top_k_probabilities * old_top_k_probabilities)
                    )
        top_k_probabilities = top_k_probabilities * tf.reduce_sum(self.beam_rift_space, axis=0)

        candidate_beams = tf.squeeze(tf.gather(top_k_probabilities, top_k_indices, axis=1))

        if candidate_beams.shape.rank < 4:
            for i in range(4 - candidate_beams.shape.rank):
                candidate_beams = tf.expand_dims(candidate_beams, i)

        return candidate_beams

    def update_memory(self, w):
        self.m_cache.assign_add(w)
        candidate_beams = self.beam_rift(self.m_cache)

        def cell(beam, memory_state):
            memory_state = memory_state[0]
            dilation_rate = conv_utils.normalize_tuple(
                1, 1, "dilation_rate"
            )
            dilation_rate = (1,) + dilation_rate

            spatial_start_dim = 1

            strides = conv_utils.normalize_tuple(
                beam.shape[2], 1, "strides", allow_zero=True
            )

            beam = tf.expand_dims(beam, spatial_start_dim)
            depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)

            output = tf.nn.depthwise_conv2d(
                beam,
                depthwise_kernel,
                strides=(1,) + strides * 2 + (1,),
                padding=conv_utils.normalize_padding("valid").upper(),
                dilations=dilation_rate,
                data_format=conv_utils.convert_data_format(
                    conv_utils.normalize_data_format("channels_last"), ndim=4
                ),
            )

            output = tf.squeeze(output, spatial_start_dim)
            memory_state = (memory_state + (output * memory_state) /
                            (tf.reduce_sum(output, axis=1) * tf.reduce_sum(memory_state, axis=1)))
            return output, [memory_state]

        _, _, memory = tf.keras.backend.rnn(
            cell,
            candidate_beams,
            [self.m]
        )

        self.m.assign(memory[0])

    def call(
            self,
            x,
            mask=None,
    ):
        scores = super().call(x, 0, mask)
        scores = tf.math.multiply(scores, self.m)

        self.update_memory(scores)

        return scores
