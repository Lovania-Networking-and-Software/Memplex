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


class RoundedNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, name="rounded_norm"):
        super().__init__(layer, name=name)

        self.norm = tf.keras.layers.UnitNormalization()

    def call(self, inputs):
        return tf.round(
            self.norm(
                self.layer(inputs)
            )
        )


class PoolingAndAverageBasedSpaceCreatorLayer(tf.keras.layers.Layer):
    def __init__(self, name="space_creator"):
        super().__init__(False, name)
        self.pooling_max = tf.keras.layers.GlobalMaxPooling1D(name="pooling_space_max", keepdims=True)
        self.pooling_average = tf.keras.layers.GlobalAveragePooling1D(
            name="pooling_space_average", keepdims=True)

        self.flatten = tf.keras.layers.Flatten(name="space_flattener")
        self.norm = tf.keras.layers.UnitNormalization(name="space_normalizer")

        self.add = RoundedNorm(tf.keras.layers.Add())
        self.average = tf.keras.layers.Average(name="space_average")
        self.concatenate = tf.keras.layers.Concatenate(name="space_concatenate")

    def call(self, inputs):
        pooled_space_max = self.norm(self.pooling_max(inputs))
        pooled_space_average = self.norm(self.pooling_average(inputs))

        average = self.average([pooled_space_max, pooled_space_average])
        inputs = self.add([pooled_space_average, pooled_space_max])

        average = self.average([average, inputs])
        inputs = self.add([average, inputs])

        average = self.average([inputs, average])
        inputs = self.add([inputs, average])

        base_space = tf.round(self.concatenate([inputs, average]))
        filtered_space = tf.boolean_mask(base_space, tf.not_equal(base_space, 0))

        return self.flatten(filtered_space)


class Labeler(tf.Module):
    def __init__(self, name="labeler"):
        super().__init__(name=name)
        self.space_creator = PoolingAndAverageBasedSpaceCreatorLayer()
        self.fourier_features = tf.keras.layers.experimental.RandomFourierFeatures(128, "gaussian", 4.,
                                                                                   False, "feauture_space_projector")
        self.linear = tf.keras.layers.PReLU(name="linear_space")

    def __call__(self, inputs):
        attention_space = self.space_creator(inputs)

        feature_space = tf.round(self.fourier_features(attention_space))

        linear_data = self.linear(feature_space)

        return linear_data
