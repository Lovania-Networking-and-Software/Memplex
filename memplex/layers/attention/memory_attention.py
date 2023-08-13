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

import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

from memplex.layers.attention.attention import Attention


class MemoryAttention(Attention):
    m = None
    mCache = None
    depthwise_kernel = None

    def __init__(
            self,
            n_heads,
            n_kv_heads,
            dim,
            max_batch_size,
            max_seq_len
    ):
        super().__init__(n_heads, n_kv_heads, dim, max_batch_size, max_seq_len)

    @tf_utils.shape_type_conversion
    def build(
            self,
            input_shape
    ):
        beam_shape = self.compute_beam_shape(input_shape)
        self.m = self.add_weight(
            "memory",
            [beam_shape[0], 1, beam_shape[2]],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            trainable=False
        )
        self.mCache = self.add_weight(
            "memory_cache",
            input_shape,
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            trainable=False
        )

        kernel_size = conv_utils.normalize_tuple(
            beam_shape[1], 1, "kernel_size"
        )
        input_dim = int(beam_shape[-1])
        depthwise_kernel_shape = kernel_size + (
            input_dim,
            1,
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=tf.keras.initializers.GlorotNormal(),
            name="depthwise_kernel",
            regularizer=None,
            constraint=None,
        )

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape))
        super().build(input_shape)

    def compute_beam_shape(self, input_shape):
        sample = tf.ones(input_shape)
        abs_shape = float(sample.shape[0] * sample.shape[1] * sample.shape[2])
        abs_shape = tf.math.abs(abs_shape // tf.math.sqrt(abs_shape))
        abs_const_val = tf.math.abs(2 * abs_shape)
        side_constraint = tf.keras.constraints.MinMaxNorm(
            -abs_const_val,
            abs_const_val,
            axis=1
        )
        sample = side_constraint(sample)
        beam_width = sample.shape[2]

        probabilities = tf.nn.log_softmax(sample)
        top_k_probabilities, top_k_indices = tf.nn.top_k(probabilities, k=beam_width)

        for _ in range(beam_width - 1):
            top_k_probabilities, top_k_indices = tf.nn.top_k(top_k_probabilities, k=1)
        candidate_beams = tf.squeeze(tf.gather(top_k_probabilities, top_k_indices, axis=1))

        if candidate_beams.shape.rank < 3:
            for i in range(3 - candidate_beams.shape.rank):
                candidate_beams = tf.expand_dims(candidate_beams, i)
        return candidate_beams.shape

    def update_memory(self, w):
        self.mCache.assign_add(w)

        abs_shape = float(self.mCache.shape[0] * self.mCache.shape[1] * self.mCache.shape[2])
        abs_shape = tf.math.abs(abs_shape // tf.math.sqrt(abs_shape))
        abs_const_val = tf.math.abs(2 * abs_shape)
        side_constraint = tf.keras.constraints.MinMaxNorm(
            -abs_const_val,
            abs_const_val,
            axis=1
        )
        m_cache = side_constraint(self.mCache)
        beam_width = self.mCache.shape[2]

        probabilities = tf.nn.log_softmax(m_cache)
        top_k_probabilities, top_k_indices = tf.nn.top_k(probabilities, k=beam_width)

        # Iterate over the beams and select the one with the highest score.
        for _ in range(beam_width - 1):
            top_k_probabilities, top_k_indices = tf.nn.top_k(top_k_probabilities, k=1)
        candidate_beams = tf.squeeze(tf.gather(top_k_probabilities, top_k_indices, axis=1))

        if candidate_beams.shape.rank < 3:
            for i in range(3 - candidate_beams.shape.rank):
                candidate_beams = tf.expand_dims(candidate_beams, i)

        dilation_rate = conv_utils.normalize_tuple(
            1, 1, "dilation_rate"
        )
        dilation_rate = (1,) + dilation_rate

        spatial_start_dim = 1

        inputs = tf.expand_dims(candidate_beams, spatial_start_dim)

        strides = conv_utils.normalize_tuple(
            candidate_beams.shape[2], 1, "strides", allow_zero=True
        )

        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)

        # Apply depthwise convolution to get outputs
        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=(1,) + strides * 2 + (1,),
            padding=conv_utils.normalize_padding("valid").upper(),
            dilations=dilation_rate,
            data_format=conv_utils.convert_data_format(
                conv_utils.normalize_data_format("channels_last"), ndim=4
            ),
        )

        # Reshape outputs to match the shape of self.m
        outputs = tf.squeeze(outputs, [spatial_start_dim])

        self.m.assign(outputs)

    def call(
            self,
            x,
            mask=None,
    ):
        # Pass 0 as start_pos since it's not used in MemoryAttention
        scores = super().call(x, 0, mask)
        scores = tf.math.multiply(scores, self.m)

        self.update_memory(scores)

        return scores
