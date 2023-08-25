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


def repeat_kv(
        x: tf.Tensor,
        n_rep: int
) -> tf.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return tf.reshape(
        tf.tile(tf.expand_dims(x, axis=3), [1, 1, 1, n_rep, 1]),
        [bs, slen, n_kv_heads * n_rep, head_dim]
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (tf.range(0, dim, 2, dtype=tf.float32)[: (dim // 2)] / dim))
    t = tf.range(end, dtype=freqs.dtype)
    freqs = tf.matmul(tf.expand_dims(t, -1), tf.expand_dims(freqs, 0))
    freqs_cis = tf.complex(tf.cos(freqs), tf.sin(freqs))
    return freqs_cis


def reshape_for_broadcast(freqs_cis: tf.Tensor, x: tf.Tensor):
    ndim = tf.rank(x)
    assert 0 <= 1 < ndim
    freqs_cis = tf.reshape(freqs_cis, (x.shape[1], x.shape[-1]))
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return tf.reshape(freqs_cis, shape)


def apply_rotary_emb(xq: tf.Tensor, xk: tf.Tensor, freqs_cis: tf.Tensor):
    xq = tf.reshape(xq, (*xq.shape[:-1], -1, 2))
    xk = tf.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_ = tf.dtypes.complex(xq, 0.0)
    xk_ = tf.dtypes.complex(xk, 0.0)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = tf.reshape(xq_ * freqs_cis, [-1, xq_.shape[-1]])
    xk_out = tf.reshape(xk_ * freqs_cis, [-1, xk_.shape[-1]])
    return tf.dtypes.cast(xq_out, xq.dtype), tf.dtypes.cast(xk_out, xk.dtype)


class Attention(tf.keras.layers.Layer):
    def __init__(
            self,
            n_heads,
            n_kv_heads,
            dim,
            max_batch_size,
            max_seq_len
    ):
        super(Attention, self).__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        self.mbs = max_batch_size
        self.msl = max_seq_len

        self.freqs_cis = precompute_freqs_cis(
            dim // n_heads, max_seq_len * 2
        )

        self.wq = tf.keras.layers.Dense(
            n_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.identity(),
        )
        self.wk = tf.keras.layers.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.identity(),
        )
        self.wv = tf.keras.layers.Dense(
            self.n_kv_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.identity(),
        )
        self.wo = tf.keras.layers.Dense(
            dim,
            use_bias=False,
            input_shape=(n_heads * self.head_dim,),
            kernel_initializer=tf.keras.initializers.identity(),
        )

        self.cache_k = tf.Variable(
            tf.zeros(
                (
                    max_batch_size,
                    max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ),
            trainable=False,
        )
        self.cache_v = tf.Variable(
            tf.zeros(
                (
                    max_batch_size,
                    max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ),
            trainable=False,
        )

    def call(
            self,
            x,
            start_pos,
            mask=None,
    ):
        bsz, seqlen, _ = x.shape.as_list()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = tf.reshape(xq, (bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = tf.reshape(xk, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        xv = tf.reshape(xv, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))

        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = tf.reshape(xq, (self.mbs, self.msl, self.n_local_kv_heads, self.head_dim,))
        xk = tf.reshape(xk, (self.mbs, self.msl, self.n_local_kv_heads, self.head_dim,))

        self.cache_k = self.cache_k.assign(xq)
        self.cache_v = self.cache_v.assign(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen].assign(xk)
        self.cache_v[:bsz, start_pos: start_pos + seqlen].assign(xv)

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = tf.transpose(xq, perm=[0, 2, 1, 3])  # (bs, n_local_heads, seqlen, head_dim)
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        scores = tf.matmul(xq, tf.transpose(keys, perm=[0, 1, 3, 2])) / tf.sqrt(
            tf.cast(self.head_dim, tf.float32))
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (bsz, seqlen, -1))
        return self.wo(output)
