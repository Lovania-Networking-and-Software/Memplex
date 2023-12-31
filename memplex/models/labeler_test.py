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

import unittest

import tensorflow as tf

from memplex import models


class LabelerTest(unittest.TestCase):
    def test_labeler(self):
        labeler = models.Labeler()
        labeler.build((3, 3, 3))
        labeler.summary()
        base_data = tf.random.uniform((3, 3, 3))
        data1 = tf.identity(base_data)
        data2 = tf.identity(base_data) + tf.random.normal((3, 3, 3), mean=0.0, stddev=0.4)
        self.assertEqual(tf.reduce_mean(labeler(data1)), tf.reduce_mean(labeler(data2)))


if __name__ == '__main__':
    unittest.main()
