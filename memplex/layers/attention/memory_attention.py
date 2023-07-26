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

from memplex.layers.attention.base_dense_attention import BaseDenseMemoryAttention


class MemoryAttention(BaseDenseMemoryAttention):
    def __init__(self, name="memory_attention"):
        super().__init__(name=name)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.score_weight: tf.Variable = self.add_weight(
            name="score_weight",
            shape=(),
            initializer="ones",
            dtype=self.dtype,
            trainable=True,
        )
        self.memory: tf.Variable = self.add_weight(
            shape=input_shape,
            name="memory_weight",
            initializer=tf.keras.initializers.Orthogonal(),
            regularizer=tf.keras.regularizers.OrthogonalRegularizer(),
            constraint=tf.keras.constraints.RadialConstraint(),
            trainable=False,
        )
        sample = tf.ones(input_shape)
        pos_shape = self._calculate_scores(sample).shape
        self.positional_scores: tf.Variable = self.add_weight(
            name="positional_scores",
            shape=pos_shape,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=self.dtype,
            trainable=True,
        )
        self._input_shape = input_shape
        self.input_spec: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
            ndim=len(input_shape))
        self.built = True

    def _calculate_probs(self, scores):
        probs = tf.nn.softmax(scores + self.positional_scores)
        self.memory.assign_add(probs * scores)
        return probs

    def _calculate_scores(self, inputs):
        q_reshaped = tf.expand_dims(self.memory, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = tf.expand_dims(inputs, axis=-3)
        scores = self.score_weight * tf.reduce_sum(
            tf.tanh(q_reshaped + k_reshaped), axis=-4
        )
        return scores
