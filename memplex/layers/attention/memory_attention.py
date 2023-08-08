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
from tensorflow.python.keras.utils import tf_utils

from memplex.layers.attention.attention import Attention


class MemoryAttention(Attention):
    m = None

    @tf_utils.shape_type_conversion
    def build(
            self,
            input_shape
    ):
        self.m = self.add_weight(
            "memory",
            [input_shape[1], input_shape[2]],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            constraint=tf.keras.constraints.MinMaxNorm(),
            trainable=False
        )
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape))
        super().build(input_shape)

    def call(
            self,
            x,
            mask=None,
    ):
        scores = super().call(
            x,
            0,
            mask
        )  # Pass 0 as start_pos since it's not used in MemoryAttention
        scores = tf.math.multiply(scores, self.m)
        beam_width = scores.shape[2]
        beam_width_reduction_rate = 0.05 * scores.shape[2]

        probabilities = tf.math.softmax(scores)
        top_k_probabilities, top_k_indices = tf.math.top_k(probabilities, k=int(beam_width))

        beam_scores = tf.reduce_sum(top_k_probabilities * tf.cast(top_k_indices, dtype=tf.float32),
                                    axis=1)

        def beam_width_range(start, end, step):
            numbers = []
            while start <= end:
                numbers.append(start)
                start += step
            return numbers

        for i in beam_width_range(beam_width, 1, -beam_width_reduction_rate):
            probabilities = tf.math.softmax(beam_scores)
            top_k_probabilities, top_k_indices = tf.math.top_k(probabilities, k=i)
            beam_scores = tf.reduce_sum(top_k_probabilities * top_k_indices, axis=1)

        self.m.assign_add(beam_scores)
        return scores
