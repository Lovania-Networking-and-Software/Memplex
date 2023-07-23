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

import sys
from collections import deque
from typing import Any, Iterator

import tensorflow as tf
from safetensors import safe_open
from safetensors.tensorflow import save_file


class Sizes:
    KB: int = 1024
    MB: int = KB * 1024
    GB: int = MB * 1024
    TB: int = GB * 1024


class Memory:
    def __init__(self, memory_size_threshold: int = 1 * Sizes.GB,
                 filename: str = "experience.safetensors"):
        self._filename = filename
        self._memory_size_threshold = memory_size_threshold
        self._capacity = memory_size_threshold
        self._queue = deque()
        self._data = dict[str, tf.Tensor]()

    def __getitem__(self, key) -> tf.Tensor:
        with tf.GradientTape():
            try:
                if key in self._data.keys():
                    self._queue.remove(key)
                    self._queue.append(key)
                res = self._data[key]
            except KeyError:
                try:
                    _pre_data = {}
                    with safe_open(self._filename, framework="tf") as f:
                        for k in f.keys():
                            _pre_data = f.get_tensor(k)
                    res = _pre_data[key]
                except FileNotFoundError or KeyError:
                    res = None
            return res

    def __setitem__(self, key, value):
        with tf.GradientTape():
            try:
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                _pre_data[key] = value
                if key in self._data.keys():
                    self._queue.remove(key)
                self._queue.append(key)
                self._data[key] = value
                if sys.getsizeof(self._queue) > self._capacity:
                    save_file(self._data, self._filename)
                    del self._data[self._queue.popleft()]
                save_file(_pre_data, self._filename)
            except FileNotFoundError:
                self._queue.append(key)
                self._data[key] = value
                if sys.getsizeof(self._queue) > self._capacity:
                    save_file(self._data, self._filename)
                    del self._data[self._queue.popleft()]
                save_file(self._data, self._filename)

    def __delitem__(self, key):
        with tf.GradientTape():
            try:
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                del _pre_data[key]
                del self._data[key]
                self._queue.remove(key)
                save_file(_pre_data, self._filename)
            except FileNotFoundError:
                del self._data[key]
                self._queue.remove(key)
                save_file(self._data, self._filename)
            except KeyError:
                raise KeyError

    def __iter__(self) -> Iterator[Any] | Any:
        try:
            with tf.GradientTape():
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                return iter(_pre_data)
        except FileNotFoundError:
            return iter(self._data)

    def __len__(self) -> int:
        try:
            with tf.GradientTape():
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                return len(_pre_data)
        except FileNotFoundError:
            return len(self._data)

    def __str__(self) -> str:
        try:
            with tf.GradientTape():
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                return str(_pre_data)
        except FileNotFoundError:
            return str(self._data)

    def __repr__(self) -> str:
        try:
            with tf.GradientTape():
                _pre_data = {}
                with safe_open(self._filename, framework="tf") as f:
                    for k in f.keys():
                        _pre_data[k] = f.get_tensor(k)
                return repr(_pre_data)
        except FileNotFoundError:
            return repr(self._data)
