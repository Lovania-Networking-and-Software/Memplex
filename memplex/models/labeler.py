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

from memplex.errors import NonTrainablePart
from memplex.layers import NonTrainablePReLU, \
    PoolingAndAverageBasedSpaceCreatorLayer


class Labeler(tf.keras.Model):
    def __init__(self, name="labeler"):
        super().__init__(name=name, trainable=False)
        self.space_creator = PoolingAndAverageBasedSpaceCreatorLayer()
        self.fourier_features = tf.keras.layers.experimental.RandomFourierFeatures(128, "gaussian",
                                                                                   4.,
                                                                                   False,
                                                                                   "feature_space_"
                                                                                   "projector")
        self.linear = NonTrainablePReLU(name="non-trainable_linear_space")

    def fit(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
    ):
        raise NonTrainablePart(self)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        if kwargs.get("training"):
            raise NonTrainablePart(self)
        space = self.space_creator(inputs)

        feature_space = tf.round(self.fourier_features(space))

        linear_data = self.linear(feature_space)

        return linear_data
