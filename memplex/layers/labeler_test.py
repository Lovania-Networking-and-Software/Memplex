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
import unittest

import numpy as np
import tensorflow as tf

from memplex import layers


class LabelerTest(unittest.TestCase):
    def test_labeler(self):
        labeler = layers.Labeler()
        base_data = np.random.rand(3, 3, 3)
        data1 = np.copy(base_data)
        data2 = np.copy(base_data)
        data2 += np.random.normal(0, 0.4, size=(3, 3, 3))
        self.assertEqual(tf.reduce_mean(labeler(data1)), tf.reduce_mean(labeler(data2)))


if __name__ == '__main__':
    unittest.main()
