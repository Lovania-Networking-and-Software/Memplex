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

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util


class BaseDenseMemoryAttention(base_layer.BaseRandomLayer):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.supports_masking = False

    def build(self, input_shape):
        if self.dropout > 0:
            super().build(input_shape)
        self.built = True

    def _calculate_scores(self, key):
        return NotImplementedError

    def _calculate_probs(self, scores):
        return NotImplementedError

    def _apply_scores(self, scores, value, training=None):
        if training is None:
            training = backend.learning_phase()
        weights = self._calculate_probs(scores)
        if self.dropout > 0:
            def dropped_weights():
                return self._random_generator.dropout(
                    weights, rate=self.dropout
                )

            weights = control_flow_util.smart_cond(
                training, dropped_weights, lambda: tf.identity(weights)
            )
        return weights * value, weights

    def call(
            self,
            inputs,
            mask=None,
            training=None,
            return_attention_scores=False,
    ):
        scores = self._calculate_scores(inputs)
        result, attention_scores = self._apply_scores(
            scores=scores, value=inputs, training=training
        )
        if return_attention_scores:
            return result, attention_scores
        return result

    def compute_mask(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tf.convert_to_tensor(q_mask)
        return None

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0])

    def get_config(self):
        config = {
            "dropout": self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
