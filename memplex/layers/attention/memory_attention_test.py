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

from memplex.layers.attention.memory_attention import MemoryAttention


class MemoryAttentionTest(unittest.TestCase):
    def test_attention(self):
        attention = MemoryAttention(1, 1, 4, 1, 1)
        data = tf.random.uniform([1, 1, 4])
        x1 = attention(data)
        x2 = attention(data)
        print(x1)
        print(x2)


if __name__ == '__main__':
    unittest.main()