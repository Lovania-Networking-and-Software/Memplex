#  Copyright 2023 Lovania Networking and Software
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ===========================================================================

import tensorflow as tf
from tensorflow.python.keras import backend, constraints, initializers, regularizers
from tensorflow.python.keras.utils import tf_utils

from memplex.errors import NonTrainablePart


class NonTrainablePReLU(tf.keras.layers.Layer):
    def __init__(
            self,
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=None,
            **kwargs
    ):
        super().__init__(trainable=False, **kwargs)
        self.supports_masking = False
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
            trainable=False,
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, **kwargs) -> tf.Tensor:
        if kwargs.get("training"):
            raise NonTrainablePart(self)
        pos = backend.relu(inputs)
        neg = -self.alpha * backend.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = {
            "alpha_initializer": initializers.serialize(self.alpha_initializer),
            "alpha_regularizer": regularizers.serialize(self.alpha_regularizer),
            "alpha_constraint": constraints.serialize(self.alpha_constraint),
            "shared_axes": self.shared_axes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class RoundedNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, name="rounded_norm"):
        super().__init__(layer, name=name)

        self.norm = tf.keras.layers.UnitNormalization()

    def call(self, inputs, **kwargs) -> tf.Tensor:
        if kwargs.get("training"):
            raise NonTrainablePart(self)
        return tf.round(
            self.norm(
                self.layer(inputs, **kwargs)
            )
        )


class PoolingAndAverageBasedSpaceCreatorLayer(tf.keras.layers.Layer):
    def __init__(self, name="space_creator"):
        super().__init__(trainable=False, name=name)
        self.pooling_max = tf.keras.layers.GlobalMaxPooling1D(name="pooling_space_max",
                                                              keepdims=True)
        self.pooling_average = tf.keras.layers.GlobalAveragePooling1D(
            name="pooling_space_average", keepdims=True)

        self.flatten = tf.keras.layers.Flatten(name="space_flattener")
        self.norm = tf.keras.layers.UnitNormalization(name="space_normalizer")

        self.add = RoundedNorm(tf.keras.layers.Add())
        self.average = tf.keras.layers.Average(name="space_average")
        self.concatenate = tf.keras.layers.Concatenate(name="space_concatenate")

    def call(self, inputs, **kwargs) -> tf.Tensor:
        if kwargs.get("training"):
            raise NonTrainablePart(self)
        pooled_space_max = self.norm(self.pooling_max(inputs))
        pooled_space_average = self.norm(self.pooling_average(inputs))

        average = self.average([pooled_space_max, pooled_space_average])
        inputs = self.add([pooled_space_average, pooled_space_max])

        average2 = self.average([average, inputs])
        inputs = self.add([average, inputs])

        average = self.average([inputs, average2])
        inputs = self.add([inputs, average2])

        base_space = tf.round(self.concatenate([inputs, average]))
        filtered_space = tf.boolean_mask(base_space, tf.not_equal(base_space, 0))

        return self.flatten(filtered_space)
