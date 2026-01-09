# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging


_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'clrs_processor'


class Processor(hk.Module):
  """Processor abstract base class."""

  def __init__(self, name: str):
    if not name.endswith(PROCESSOR_TAG):
      name = name + '_' + PROCESSOR_TAG
    super().__init__(name=name)

  @abc.abstractmethod
  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> Tuple[_Array, Optional[_Array]]:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step as a 2-tuple of (node, edge)
      embeddings. The edge embeddings can be None.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False

class RT(Processor):
  """
  Relational Transformers (Diao et al., 2023). 
  https://github.com/CameronDiao/relational-transformer/blob/master/clrs/_src/processors.py
  """

  def __init__(
          self,
          nb_layers: int,
          nb_heads: int,
          vec_size: int,
          node_hid_size: int,
          edge_hid_size_1: int,
          edge_hid_size_2: int,
          graph_vec: str,
          disable_edge_updates: bool,
          name: str = 'rt',
  ):
      super().__init__(name=name)
      assert graph_vec in ['att', 'core', 'cat']
      self.nb_layers = nb_layers
      self.nb_heads = nb_heads
      self.graph_vec = graph_vec
      self.disable_edge_updates = disable_edge_updates

      self.node_vec_size = vec_size
      self.node_hid_size = node_hid_size
      self.edge_vec_size = vec_size
      self.edge_hid_size_1 = edge_hid_size_1
      self.edge_hid_size_2 = edge_hid_size_2
      self.global_vec_size = vec_size

      self.tfm_dropout_rate = 0.0

  def __call__(
          self,
          node_fts: _Array,
          edge_fts: _Array,
          graph_fts: _Array,
          adj_mat: _Array,
          hidden: _Array,
          **unused_kwargs,
  ) -> _Array:
      N = node_fts.shape[-2]
      node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
      edge_tensors = jnp.concatenate([edge_fts, unused_kwargs.get('e_hidden')], axis=-1)
      if self.graph_vec == 'core':
          graph_tensors = jnp.concatenate([graph_fts, unused_kwargs.get('g_hidden')], axis=-1)
      else:
          graph_tensors = graph_fts

      if self.graph_vec == 'cat':
          expanded_graph_tensors = jnp.tile(jnp.expand_dims(graph_tensors, -2), (1, N, 1))
          node_tensors = jnp.concatenate([node_tensors, expanded_graph_tensors], axis=-1)
          expanded_graph_tensors = jnp.tile(jnp.expand_dims(graph_tensors, (-2, -3)), (1, N, N, 1))
          edge_tensors = jnp.concatenate([edge_tensors, expanded_graph_tensors], axis=-1)

      node_enc = hk.Linear(self.node_vec_size)
      edge_enc = hk.Linear(self.edge_vec_size)
      if self.graph_vec == 'core':
          global_enc = hk.Linear(self.global_vec_size)

      node_tensors = node_enc(node_tensors)
      edge_tensors = edge_enc(edge_tensors)
      if self.graph_vec == 'core':
          graph_tensors = global_enc(graph_tensors)
          expanded_graph_tensors = jnp.expand_dims(graph_tensors, 1)
          node_tensors = jnp.concatenate([expanded_graph_tensors, node_tensors], axis=-2)
          edge_tensors = jnp.pad(edge_tensors, [(0, 0), (1, 0), (1, 0), (0, 0)], mode='constant', constant_values=0.)

      layers = []
      for l in range(self.nb_layers):
          layers.append(Basic_RT(self.nb_heads,
                                  self.graph_vec,
                                  self.disable_edge_updates,
                                  self.node_vec_size,
                                  self.node_hid_size,
                                  self.edge_vec_size,
                                  self.edge_hid_size_1,
                                  self.edge_hid_size_2,
                                  self.tfm_dropout_rate,
                                  name='{}_layer{}'.format(self.name, l)))
      for layer in layers:
          node_tensors, edge_tensors = layer(node_tensors, edge_tensors, graph_tensors, adj_mat, hidden)

      if self.graph_vec == 'core':
          out_nodes = node_tensors[:, 1:, :]
          out_edges = edge_tensors[:, 1:, 1:, :]
          out_graph = node_tensors[:, 0, :]
      else:
          out_nodes = node_tensors
          out_edges = edge_tensors
          out_graph = graph_tensors

      return out_nodes, out_edges, out_graph if self.graph_vec == 'core' else None


class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GAT inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATFull(GAT):
  """Graph Attention Network with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GATv2 inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


class GATv2FullD2(GATv2):
  """Graph Attention Network v2 with full adjacency matrix and D2 symmetry."""

  def d2_forward(self,
                 node_fts: List[_Array],
                 edge_fts: List[_Array],
                 graph_fts: List[_Array],
                 adj_mat: _Array,
                 hidden: _Array,
                 **unused_kwargs) -> List[_Array]:
    num_d2_actions = 4

    d2_inverses = [
        0, 1, 2, 3  # All members of D_2 are self-inverses!
    ]

    d2_multiply = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    assert len(node_fts) == num_d2_actions
    assert len(edge_fts) == num_d2_actions
    assert len(graph_fts) == num_d2_actions

    ret_nodes = []
    adj_mat = jnp.ones_like(adj_mat)

    for g in range(num_d2_actions):
      emb_values = []
      for h in range(num_d2_actions):
        gh = d2_multiply[d2_inverses[g]][h]
        node_features = jnp.concatenate(
            (node_fts[g], node_fts[gh]),
            axis=-1)
        edge_features = jnp.concatenate(
            (edge_fts[g], edge_fts[gh]),
            axis=-1)
        graph_features = jnp.concatenate(
            (graph_fts[g], graph_fts[gh]),
            axis=-1)
        cell_embedding = super().__call__(
            node_fts=node_features,
            edge_fts=edge_features,
            graph_fts=graph_features,
            adj_mat=adj_mat,
            hidden=hidden
        )
        emb_values.append(cell_embedding[0])
      ret_nodes.append(
          jnp.mean(jnp.stack(emb_values, axis=0), axis=0)
      )

    return ret_nodes


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)

def get_falr1_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Only get node information. Ignore edges (f1)"""
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_3 = t_3(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )

# Small MLP with nonlinearity for memory effect
def non_linear_memory_block(triplets, nb_triplet_fts):
    mem_layer1 = hk.Linear(nb_triplet_fts)
    mem_layer2 = hk.Linear(nb_triplet_fts)
    t1 = mem_layer1(triplets)
    t1 = jax.nn.relu(t1)
    t1 = mem_layer2(t1)
    return t1

# Simplified LSTM memory block
def lstm_memory_block(triplets, nb_triplet_fts):
  orig_shape = triplets.shape
  flat_triplets = jnp.reshape(triplets, (-1, orig_shape[-1]))
  lstm = hk.LSTM(nb_triplet_fts)

  # Use zeros as initial state
  state = lstm.initial_state(flat_triplets.shape[0])

  # Single LSTM step (treat input as one timestep)
  output, _ = lstm(flat_triplets, state)
  output = jnp.reshape(output, orig_shape[:-1] + (nb_triplet_fts,))
  return output

# Simplified GRU memory block
def gru_memory_block(triplets, nb_triplet_fts):
  orig_shape = triplets.shape
  flat_triplets = jnp.reshape(triplets, (-1, orig_shape[-1]))
  gru = hk.GRU(nb_triplet_fts)

  # Use zeros as initial state
  state = gru.initial_state(flat_triplets.shape[0])

  # Single GRU step (treat input as one timestep)
  output, _ = gru(flat_triplets, state)
  output = jnp.reshape(output, orig_shape[:-1] + (nb_triplet_fts,))
  return output

def get_falr2_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Only get node information. Ignore edges (f1)"""

  tri_1 = lstm_memory_block(z, nb_triplet_fts)
  tri_2 = lstm_memory_block(z, nb_triplet_fts)
  tri_e_1 = lstm_memory_block(edge_fts, nb_triplet_fts)
  tri_g = lstm_memory_block(graph_fts, nb_triplet_fts)

  '''
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_g = t_g(graph_fts)
  '''

  return (
      jnp.expand_dims(tri_1, axis=(1))    +  # (B, 1, N, H)
      jnp.expand_dims(tri_2, axis=(2))    +  # (B, N, 1, H)
      tri_e_1                             +  # (B, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2))    # (B, 1, 1, H)
  ) 

def get_falr3_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(1))    +  # (B, 1, N, H)
      jnp.expand_dims(tri_2, axis=(2))    +  # (B, N, 1, H)
      tri_e_1                             +  # (B, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2))    # (B, 1, 1, H)
  ) 

def get_falr4_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  tri_1 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))(z)
  tri_2 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(z)
  tri_3 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.RandomNormal(stddev=0.05))(z)
  tri_4 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.Orthogonal())(z)

  tri_e_1 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)

  tri_g1 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))(graph_fts)
  tri_g2 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(graph_fts)
  tri_g3 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.RandomNormal(stddev=0.05))(graph_fts)
  tri_g4 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.Orthogonal())(graph_fts)

  tri_1_exp = jnp.expand_dims(tri_1, axis=(1))    # (B, 1, N, H)
  tri_2_exp = jnp.expand_dims(tri_2, axis=(2))    # (B, N, 1, H)
  tri_3_exp = jnp.expand_dims(tri_3, axis=(1))    # (B, 1, N, H)
  tri_4_exp = jnp.expand_dims(tri_4, axis=(2))    # (B, N, 1, H)
  
  tri_g_exp1 = jnp.expand_dims(tri_g1, axis=(1, 2)) # (B, 1, 1, H)
  tri_g_exp2 = jnp.expand_dims(tri_g2, axis=(1, 2)) # (B, 1, 1, H)
  tri_g_exp3 = jnp.expand_dims(tri_g3, axis=(1, 2)) # (B, 1, 1, H)
  tri_g_exp4 = jnp.expand_dims(tri_g4, axis=(1, 2)) # (B, 1, 1, H)

  # Combine triplet and graph features using weighted sum and nonlinearities for more expressiveness
  msg = (
      tri_1_exp +
      tri_2_exp +
      tri_3_exp +
      tri_4_exp +
      tri_e_1 +
      tri_g_exp1 +
      tri_g_exp2 +
      tri_g_exp3 +
      tri_g_exp4
  )

  return msg

def get_falr5_msgs(node_fts, hidden, edge_fts, graph_fts, nb_triplet_fts):
  tri_1 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))(node_fts)
  tri_2 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(hidden)
  tri_3 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.RandomNormal(stddev=0.05))(node_fts)
  tri_4 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.Orthogonal())(hidden)

  tri_e_1 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)

  tri_g1 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))(graph_fts)
  tri_g2 = hk.Linear(nb_triplet_fts, with_bias=True, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(graph_fts)

  tri_1_exp = jnp.expand_dims(tri_1, axis=(1))    # (B, 1, N, H)
  tri_2_exp = jnp.expand_dims(tri_2, axis=(2))    # (B, N, 1, H)
  tri_3_exp = jnp.expand_dims(tri_3, axis=(1))    # (B, 1, N, H)
  tri_4_exp = jnp.expand_dims(tri_4, axis=(2))    # (B, N, 1, H)
  
  tri_g_exp1 = jnp.expand_dims(tri_g1, axis=(1, 2)) # (B, 1, 1, H)
  tri_g_exp2 = jnp.expand_dims(tri_g2, axis=(1, 2)) # (B, 1, 1, H)

  # Combine triplet and graph features using weighted sum and nonlinearities for more expressiveness
  msg = (
      tri_1_exp +
      tri_2_exp +
      tri_3_exp +
      tri_4_exp +
      tri_e_1 +
      tri_g_exp1 +
      tri_g_exp2
  )
  return msg

def get_falr6_msgs(node_fts, hidden, edge_fts, graph_fts, nb_triplet_fts):
  tri_n_1 = hk.Linear(nb_triplet_fts, with_bias=True)(node_fts)
  tri_n_2 = hk.Linear(nb_triplet_fts, with_bias=True)(node_fts)
  
  tri_h_1 = hk.Linear(nb_triplet_fts, with_bias=True)(hidden)
  tri_h_2 = hk.Linear(nb_triplet_fts, with_bias=True)(hidden)

  tri_e_1 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)
  tri_e_2 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)
  # tri_e_3 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)

  tri_g1 = hk.Linear(nb_triplet_fts, with_bias=True)(graph_fts)
  tri_g2 = hk.Linear(nb_triplet_fts, with_bias=True)(graph_fts)

  tri_n1_exp = jnp.expand_dims(tri_n_1, axis=(1))    # (B, 1, N, H)
  tri_n2_exp = jnp.expand_dims(tri_n_2, axis=(2))    # (B, N, 1, H)

  tri_h1_exp = jnp.expand_dims(tri_h_1, axis=(1))    # (B, 1, N, H)
  tri_h2_exp = jnp.expand_dims(tri_h_2, axis=(2))    # (B, N, 1, H)
  
  tri_g_exp1 = jnp.expand_dims(tri_g1, axis=(1, 2)) # (B, 1, 1, H)
  tri_g_exp2 = jnp.expand_dims(tri_g2, axis=(1, 2)) # (B, 1, 1, H)

  # Combine triplet and graph features using weighted sum and nonlinearities for more expressiveness
  msg = (
      tri_n1_exp + tri_n2_exp +
      tri_h1_exp + tri_h2_exp +
      tri_e_1 + tri_e_2 + #tri_e_3 +
      tri_g_exp1 + tri_g_exp2
  )

  return msg

def get_falr7_msgs(node_fts, hidden, edge_fts, graph_fts, nb_triplet_fts):
  tri_n_1 = hk.Linear(nb_triplet_fts, with_bias=True)(node_fts)  # (B, N, H)
  tri_h_1 = hk.Linear(nb_triplet_fts, with_bias=True)(hidden)    # (B, N, H)
  tri_e_1 = hk.Linear(nb_triplet_fts, with_bias=True)(edge_fts)  # (B, N, N, H)
  tri_g_1 = hk.Linear(nb_triplet_fts, with_bias=True)(graph_fts) # (B, H)

  #tri_n_1 = non_linear_memory_block(node_fts, nb_triplet_fts)
  tri_n_2 = non_linear_memory_block(node_fts, nb_triplet_fts)

  #tri_h_1 = non_linear_memory_block(hidden, nb_triplet_fts)
  tri_h_2 = non_linear_memory_block(node_fts, nb_triplet_fts)

  #tri_e_1 = non_linear_memory_block(edge_fts, nb_triplet_fts)
  tri_e_2 = non_linear_memory_block(edge_fts, nb_triplet_fts)

  #tri_g_1 = non_linear_memory_block(graph_fts, nb_triplet_fts)
  tri_g_2 = non_linear_memory_block(graph_fts, nb_triplet_fts)

  B, N, H = tri_n_1.shape

  tri_n_1_exp = jnp.expand_dims(tri_n_1, axis=(1))
  tri_n_2_exp = jnp.expand_dims(tri_n_2, axis=(1))

  tri_h_1_exp = jnp.expand_dims(tri_h_1, axis=(2))
  tri_h_2_exp = jnp.expand_dims(tri_h_2, axis=(2))

  tri_g_1_exp = jnp.expand_dims(tri_g_1, axis=(1, 2))
  tri_g_2_exp = jnp.expand_dims(tri_g_2, axis=(1, 2))

  msg = (
      tri_n_1_exp +
      tri_n_2_exp +
      tri_h_1_exp +
      tri_h_2_exp +
      tri_e_1 +
      tri_e_2 +
      tri_g_1_exp +
      tri_g_2_exp
  )

  return msg


def get_falr8_msgs(node_fts, hidden, edge_fts, graph_fts, nb_triplet_fts):
  """Only get node information. Ignore edges (f1)"""

  t_n_1 = hk.Linear(nb_triplet_fts)
  t_n_2 = hk.Linear(nb_triplet_fts)
  t_n_3 = hk.Linear(nb_triplet_fts)
  t_h_1 = hk.Linear(nb_triplet_fts)
  t_h_2 = hk.Linear(nb_triplet_fts)
  t_h_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_n_1 = t_n_1(node_fts)
  tri_n_2 = t_n_2(node_fts)
  tri_n_3 = t_n_3(node_fts)
  tri_h_1 = t_h_1(hidden)
  tri_h_2 = t_h_2(hidden)
  tri_h_3 = t_h_3(hidden)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_n_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_n_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_n_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_h_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_h_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_h_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )


##############################################################
##############################################################

class FALR1(Processor):
  """f1 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f1',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      triplets = get_falr1_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)
      #tri_msgs = jnp.max(triplets, axis=1) + jnp.max(triplets, axis=2) + jnp.max(triplets, axis=3)  # (B, N, N, H)
      tri_msgs = jnp.average(triplets, axis=1)  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(self.activation(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)

    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)

    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      gate = self.gated_activation(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      #gate = self.gated_activation(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))

      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars

class F1(FALR1):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class FALR2(Processor):
  """f2 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f2',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)
    
    tri_msgs = None

    if self.use_triplets:
      triplets = get_falr2_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts) 
      
      #simple memory block
      tri_msgs = lstm_memory_block(triplets, self.nb_triplet_fts)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + 
        jnp.expand_dims(msg_2, axis=2) +
        msg_e + 
        jnp.expand_dims(msg_g, axis=(1, 2))
    )

    msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    #if self.activation is not None:
    #  ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      gate = self.gated_activation(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F2(FALR2):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)
  

class FALR3(Processor):
  """f3 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f3',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)
    
    tri_msgs = None

    if self.use_triplets:
      tri_msgs = get_falr3_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts) 

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + 
        jnp.expand_dims(msg_2, axis=2) +
        msg_e + 
        jnp.expand_dims(msg_g, axis=(1, 2))
    )

    msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    #if self.activation is not None:
    #  ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      gate = self.gated_activation(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F3(FALR3):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)
  
class FALR4(Processor):
  """f4 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f4',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    
    # Increase data variability by using different initializations and activations for each message component
    msg_1 = hk.Linear(self.mid_size, with_bias=True, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))(z)
    msg_2 = hk.Linear(self.mid_size, with_bias=True, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'))(z)
    msg_e = hk.Linear(self.mid_size, with_bias=True)(edge_fts)
    msg_g = hk.Linear(self.mid_size, with_bias=True)(graph_fts)
    
    tri_msgs = None

    if self.use_triplets:
      tri_msgs = get_falr4_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts) 

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msg_1_exp = jnp.expand_dims(msg_1, axis=(1))    # (B, 1, N, H)
    msg_2_exp = jnp.expand_dims(msg_2, axis=(2))    # (B, N, 1, H)
    msg_g_exp = jnp.expand_dims(msg_g, axis=(1, 2)) # (B, 1, 1, H)

    msgs = (
        msg_1_exp + msg_2_exp + msg_e + msg_g_exp
    )

    msgs = self.reduction(msgs, axis=1)
    #msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = hk.Linear(self.out_size, with_bias=True)(z)
    h_2 = hk.Linear(self.out_size, with_bias=True)(msgs)

    ret = h_1 + h_2

    #if self.activation is not None:
    #  ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size, with_bias=True)
      gate2 = hk.Linear(self.out_size, with_bias=True)
      gate3 = hk.Linear(self.out_size, with_bias=True, b_init=hk.initializers.Constant(-3))

      gate = self.gated_activation(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F4(FALR4):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)
  

class FALR5(Processor):
  """f5 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f5',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    #z = jnp.concatenate([node_fts, hidden], axis=-1)
    msg_11 = hk.Linear(self.mid_size, with_bias=True)(node_fts)
    msg_12 = hk.Linear(self.mid_size, with_bias=True)(node_fts)
    msg_21 = hk.Linear(self.mid_size, with_bias=True)(hidden)
    msg_22 = hk.Linear(self.mid_size, with_bias=True)(hidden)
    msg_e = hk.Linear(self.mid_size, with_bias=True)(edge_fts)
    msg_g = hk.Linear(self.mid_size, with_bias=True)(graph_fts)

    tri_msgs = get_falr5_msgs(node_fts, hidden, edge_fts, graph_fts, self.nb_triplet_fts) 

    if self.activation is not None:
      tri_msgs = self.activation(tri_msgs)

    msg_11_exp = jnp.expand_dims(msg_11, axis=(1))    # (B, 1, N, H)
    msg_12_exp = jnp.expand_dims(msg_12, axis=(2))    # (B, 1, N, H)
    msg_21_exp = jnp.expand_dims(msg_21, axis=(1))    # (B, N, 1, H)
    msg_22_exp = jnp.expand_dims(msg_22, axis=(2))    # (B, N, 1, H)
    msg_g_exp = jnp.expand_dims(msg_g, axis=(1, 2)) # (B, 1, 1, H)

    msgs = (
        msg_11_exp + msg_12_exp + msg_21_exp + msg_22_exp + msg_e + msg_g_exp
    )

    msgs = self.reduction(msgs, axis=1)

    #h_1 = hk.Linear(self.out_size, with_bias=True)(node_fts)
    #h_2 = hk.Linear(self.out_size, with_bias=True)(hidden)
    #h_3 = hk.Linear(self.out_size, with_bias=True)(msgs)
    #ret = h_1 + h_2 + h_3

    ret = hk.Linear(self.out_size, with_bias=True)(msgs)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F5(FALR5):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class FALR6(Processor):
  """f6 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f6',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    #z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)
    o3 = hk.Linear(self.out_size)

    msg_n_1 = m_1(node_fts)
    msg_h_1 = m_2(hidden)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      tri_msgs = get_falr6_msgs(node_fts, hidden, edge_fts, graph_fts, self.nb_triplet_fts)
      #tri_msgs = jnp.average(triplets, axis=1)  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_n_1, axis=1) + jnp.expand_dims(msg_h_1, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2))
    )

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(self.activation(msgs))


    msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
    #msgs = self.reduction(msgs, axis=1)

    h_1 = o1(node_fts)
    h_2 = o2(hidden)
    h_3 = o3(msgs)

    ret = h_1 + h_2 + h_3

    #if self.activation is not None:
    #  ret = self.activation(ret)

    if self.gated:
      # Improved gating: use LayerNorm, richer interaction, and optional residual
      gate_n = hk.Linear(self.out_size)
      gate_h = hk.Linear(self.out_size)
      gate_m = hk.Linear(self.out_size)
      gate_o = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      # Concatenate all sources for gating
      gate_input = jnp.concatenate([
          gate_n(node_fts),
          gate_h(hidden),
          gate_m(msgs)
      ], axis=-1)

      # Normalization for stability
      if self.use_ln:
        ln_gate = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        gate_input = ln_gate(gate_input)

      gate = self.gated_activation(gate_o(jax.nn.relu(gate_input)))

      # Residual connection for better gradient flow
      ret = ret * gate + hidden * (1 - gate) + ret * (1 - gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F6(FALR6):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)
  
class FALR7(Processor):
  """f7 code"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      name: str = 'f7',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    #z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_n_1 = hk.Linear(self.mid_size)
    m_n_2 = hk.Linear(self.mid_size)
    m_h_1 = hk.Linear(self.mid_size)
    m_h_2 = hk.Linear(self.mid_size)
    m_e_1 = hk.Linear(self.mid_size)
    m_e_2 = hk.Linear(self.mid_size)
    m_g_1 = hk.Linear(self.mid_size)
    m_g_2 = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)
    o3 = hk.Linear(self.out_size)

    msg_n_1 = m_n_1(node_fts)
    msg_n_2 = m_n_2(node_fts)
    msg_h_1 = m_h_1(hidden)
    msg_h_2 = m_h_2(hidden)
    msg_e_1 = m_e_1(edge_fts)
    msg_e_2 = m_e_2(edge_fts)
    msg_g_1 = m_g_1(graph_fts)
    msg_g_2 = m_g_2(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      triplets = get_falr7_msgs(node_fts, hidden, edge_fts, graph_fts, self.nb_triplet_fts)

      if self.activation is not None:
        tri_msgs = self.activation(triplets)


    B, N, H = msg_n_1.shape

    msgs = (
        jnp.expand_dims(msg_n_1, axis=1) + 
        jnp.expand_dims(msg_h_1, axis=2) +
        msg_e_1 + 
        jnp.expand_dims(msg_g_1, axis=(1, 2))
    )

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(self.activation(msgs))


    msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
    #msgs = self.reduction(msgs, axis=(1))

    h_1 = o1(node_fts)
    h_2 = o2(hidden)
    h_3 = o3(msgs)

    #print("h_1", h_1.shape)
    #print("h_2", h_2.shape)
    #print("h_3", h_3.shape)

    # Add attention mechanism over node and hidden features before combining
    # Compute attention scores
    '''
    att_n = hk.Linear(1)(node_fts)  # (B, N, 1)
    att_h = hk.Linear(1)(hidden)    # (B, N, 1)
    att_scores = jnp.concatenate([att_n, att_h], axis=-1)  # (B, N, 2)
    att_weights = jax.nn.softmax(att_scores, axis=-1)      # (B, N, 2)

    # Weighted sum for node and hidden features
    node_att = att_weights[..., 0:1] * h_1 + att_weights[..., 1:2] * h_2  # (B, N, H)

    ret = node_att + h_3
    '''
    ret = h_1 + h_2 + h_3

    #if self.activation is not None:
    #  ret = self.activation(ret)

    if self.gated:
      # Improved gating: use LayerNorm, richer interaction, and optional residual
      gate_n = hk.Linear(self.out_size)
      gate_h = hk.Linear(self.out_size)
      gate_m = hk.Linear(self.out_size)
      gate_g = hk.Linear(self.out_size)
      gate_o = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      # Concatenate all sources for gating
      gate_input = jnp.concatenate([
          gate_n(node_fts),
          gate_h(hidden),
          gate_m(msgs)
      ], axis=-1)

      # Normalization for stability
      if self.use_ln:
        ln_gate = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        gate_input = ln_gate(gate_input)

      gate = self.gated_activation(gate_o(jax.nn.relu(gate_input)))

      # Residual connection for better gradient flow
      ret = (ret * gate) + (hidden * gate) + (hidden * (1 - gate)) + (ret * (1 - gate))

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F7(FALR7):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def multihead_attention_block(memory_size, out_size, fts, axis, memory):
  mha = hk.MultiHeadAttention(
    num_heads=memory_size,
    key_size=out_size // memory_size,
    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
  )

  mem_input = jnp.mean(fts, axis=axis, keepdims=True)  # (B, 1, H)

  # Attend mem_input (query) to memory (key, value)
  memory = mha(query=mem_input, key=memory, value=memory) #(B, memory_size, H), mem_input: (B, 1, H)
  
  # Generate mem_context to edge features
  mem_context = jnp.mean(memory, axis=1, keepdims=True)  # (B, 1, H)

  return memory, mem_context


def lstm_block(ini_state, memory_size, out_size, fts, axis, memory):
  # Use an LSTM cell to update memory
  lstm = hk.LSTM(out_size)
  mem_input = jnp.mean(fts, axis=axis, keepdims=True)  # (B, 1, H)

  mem_state = lstm.initial_state(ini_state)

  new_memory = []
  for i in range(memory_size):
    mem_out, mem_state = lstm(mem_input, mem_state)
    new_memory.append(mem_out)

  memory = jnp.stack(new_memory, axis=axis)  # (B, memory_size, H)

  # Generate mem_context to edge features
  mem_context = jnp.mean(memory, axis=axis, keepdims=True)  # (B, 1, H)
  
  return memory, mem_context

def gru_block(ini_state, memory_size, out_size, fts, axis, memory):
  # Use a GRU cell to update memory sequentially for each batch
  gru = hk.GRU(out_size)

  mem_input = jnp.mean(fts, axis=1)  # (B, H)
  mem_state = gru.initial_state(ini_state)

  new_memory = []
  for i in range(memory_size):
    mem_out, mem_state = gru(mem_input, mem_state)
    new_memory.append(mem_out)
  memory = jnp.stack(new_memory, axis=axis)  # (B, memory_size, H)

    # Generate mem_context to edge features
  mem_context = jnp.mean(memory, axis=axis, keepdims=True)  # (B, 1, H)
  
  return memory, mem_context

class FALR8(Processor):
  """f7 code with memory block (f8)"""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      gated: bool = False,
      gated_activation: Optional[_Fn] = jax.nn.sigmoid,
      memory_type: Optional[str] = None,
      memory_size: Optional[int] = None,
      name: str = 'f8',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated
    self.gated_activation = gated_activation
    self.memory_type = memory_type
    self.memory_size = memory_size

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      memory_n: Optional[_Array] = None,
      memory_h: Optional[_Array] = None,
      memory_e: Optional[_Array] = None,
      memory_g: Optional[_Array] = None,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step with memory."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n) #hints

    # Memory block: initialize or update memory
    if self.memory_size is None or self.memory_type is None:
      memory = None
    else:
      # Initialize memory if not provided
      if memory_e is None or memory_n is None or memory_h is None:
        memory_n = jnp.zeros((b, self.memory_size, self.out_size))
        memory_h = jnp.zeros((b, self.memory_size, self.out_size))
        memory_e = jnp.zeros((b, self.memory_size, self.out_size))
        memory_g = jnp.zeros((b, self.memory_size, self.out_size))

        memory = (memory_n, memory_h, memory_e, memory_g)
      else:
        memory_n, memory_h, memory_e, memory_g = memory

      if self.memory_type == 'gru':
        # Use a GRU cell to update memory sequentially for each batch
        memory_n, mem_context = gru_block(b, self.memory_size, self.out_size, node_fts, 1, memory_n)
        node_fts = node_fts + mem_context

      elif self.memory_type == 'lstm':
        # Use an LSTM cell to update memory
        memory_n, mem_context = lstm_block(b, self.memory_size, self.out_size, node_fts, 1, memory_n)
        node_fts = node_fts + mem_context

      elif self.memory_type == 'mha': #multi-head attention
        # Use a simple MultiHeadAttention block to update memory
        memory_n, mem_n_context = multihead_attention_block(self.memory_size, self.out_size, node_fts, 1, memory_n)
        memory_h, mem_h_context = multihead_attention_block(self.memory_size, self.out_size, hidden, 1, memory_h)
        memory_e, mem_e_context = multihead_attention_block(self.memory_size, self.out_size, edge_fts, (1,2), memory_e)
        memory_g, mem_g_context = multihead_attention_block(self.memory_size, self.out_size, graph_fts, None, memory_g)


        node_fts = node_fts + mem_n_context
        hidden = hidden + mem_h_context
        edge_fts = edge_fts + mem_e_context
        graph_fts = graph_fts + mem_g_context

        ######################
        #se usar node_fts, lembrar de atualizar a passagem de parâmetros (return da função)
        ######################

    m_n_1 = hk.Linear(self.mid_size)
    m_n_2 = hk.Linear(self.mid_size)
    m_h_1 = hk.Linear(self.mid_size)
    m_h_2 = hk.Linear(self.mid_size)
    m_e_1 = hk.Linear(self.mid_size)
    m_e_2 = hk.Linear(self.mid_size)
    m_g_1 = hk.Linear(self.mid_size)
    m_g_2 = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)
    o3 = hk.Linear(self.out_size)

    msg_n_1 = m_n_1(node_fts)
    msg_n_2 = m_n_2(node_fts)
    msg_h_1 = m_h_1(hidden)
    msg_h_2 = m_h_2(hidden)
    msg_e_1 = m_e_1(edge_fts)
    msg_e_2 = m_e_2(edge_fts)
    msg_g_1 = m_g_1(graph_fts)
    msg_g_2 = m_g_2(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      triplets = get_falr8_msgs(node_fts, hidden, edge_fts, graph_fts, self.nb_triplet_fts)
      
      ot = hk.Linear(self.out_size)
      tri_msgs = ot(jnp.max(triplets, axis=1))  # (B, N, N, H)   

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    B, N, H = msg_n_1.shape

    msgs = (
        jnp.expand_dims(msg_n_1, axis=1) + 
        jnp.expand_dims(msg_h_1, axis=2) +
        msg_e_1 + 
        jnp.expand_dims(msg_g_1, axis=(1, 2))
    )

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(self.activation(msgs))

    msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(node_fts)
    h_2 = o2(hidden)
    h_3 = o3(msgs)

    ret = h_1 + h_2 + h_3

    if self.gated:
      # Improved gating: use LayerNorm, richer interaction, and optional residual
      gate_n = hk.Linear(self.out_size)
      gate_h = hk.Linear(self.out_size)
      gate_m = hk.Linear(self.out_size)
      gate_g = hk.Linear(self.out_size)
      gate_o = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))

      # Concatenate all sources for gating
      gate_input = jnp.concatenate([
          gate_n(node_fts),
          gate_h(hidden),
          gate_m(msgs)
      ], axis=-1)

      # Normalization for stability
      if self.use_ln:
        ln_gate = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        gate_input = ln_gate(gate_input)

      gate = self.gated_activation(gate_o(jax.nn.relu(gate_input)))

      # Residual connection for better gradient flow
      ret = (ret * gate) + (hidden * gate) + (hidden * (1 - gate)) + (ret * (1 - gate))

    # Return memory as additional output if used
    if self.memory_size is not None:
      memory = (memory_n, memory_h, memory_e, memory_g)
      return ret, tri_msgs, memory   # tri_msgs and memory
    else:
      return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class F8(FALR8):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)

##############################################################
##############################################################

def get_triplet_msgs(z, edge_fts, graph_fts, nb_triplet_fts):
  """Triplet messages, as done by Dudzik and Velickovic (2022)."""
  t_1 = hk.Linear(nb_triplet_fts)
  t_2 = hk.Linear(nb_triplet_fts)
  t_3 = hk.Linear(nb_triplet_fts)
  t_e_1 = hk.Linear(nb_triplet_fts)
  t_e_2 = hk.Linear(nb_triplet_fts)
  t_e_3 = hk.Linear(nb_triplet_fts)
  t_g = hk.Linear(nb_triplet_fts)

  tri_1 = t_1(z)
  tri_2 = t_2(z)
  tri_3 = t_3(z)
  tri_e_1 = t_e_1(edge_fts)
  tri_e_2 = t_e_2(edge_fts)
  tri_e_3 = t_e_3(edge_fts)
  tri_g = t_g(graph_fts)

  return (
      jnp.expand_dims(tri_1, axis=(2, 3))    +  #   (B, N, 1, 1, H)
      jnp.expand_dims(tri_2, axis=(1, 3))    +  # + (B, 1, N, 1, H)
      jnp.expand_dims(tri_3, axis=(1, 2))    +  # + (B, 1, 1, N, H)
      jnp.expand_dims(tri_e_1, axis=3)       +  # + (B, N, N, 1, H)
      jnp.expand_dims(tri_e_2, axis=2)       +  # + (B, N, 1, N, H)
      jnp.expand_dims(tri_e_3, axis=1)       +  # + (B, 1, N, N, H)
      jnp.expand_dims(tri_g, axis=(1, 2, 3))    # + (B, 1, 1, 1, H)
  )  


class PGN(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      use_triplets: bool = False,
      nb_triplet_fts: int = 8,
      use_mean_triplet: bool = False,
      gated: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln
    self.use_triplets = use_triplets
    self.use_mean_triplet = use_mean_triplet
    self.nb_triplet_fts = nb_triplet_fts
    self.gated = gated

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    tri_msgs = None

    if self.use_triplets:
      # Triplet messages, as done by Dudzik and Velickovic (2022)
      triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)
      
      o3 = hk.Linear(self.out_size)
      tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

      if self.activation is not None:
        tri_msgs = self.activation(tri_msgs)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))

    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)

    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    if self.gated:
      gate1 = hk.Linear(self.out_size)
      gate2 = hk.Linear(self.out_size)
      gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
      gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
      ret = ret * gate + hidden * (1-gate)

    return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


class DeepSets(PGN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    assert adj_mat.ndim == 3
    adj_mat = jnp.ones_like(adj_mat) * jnp.eye(adj_mat.shape[-1])
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


class MemNetMasked(Processor):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      use_ln: bool = False,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      use_ln: whether to use layer normalisation in the model.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    self._use_ln = use_ln
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MemNet inference step."""

    del hidden
    node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                         axis=1)
    edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                              ((0, 0), (0, 1), (0, 1), (0, 0)))
    nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                               edge_fts_padded)

    # Broadcast hidden state corresponding to graph features across the nodes.
    nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
    return nxt_hidden, None  # pytype: disable=bad-return-type  # numpy-scalars

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory_embeddings = jnp.pad(
          memory_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
           (0, 0), (0, 0)))
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output_embeddings = jnp.pad(
          output_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
           (0, 0), (0, 0)))
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

    if self._use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class MemNetFull(MemNetMasked):
  """Memory Networks with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None,
                          **kwargs) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_triplet_fts: How many triplet features to compute.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  #reduction
  reduction = jnp.max #default
  
  if(kwargs['reduction'] == 'average'):
    reduction = jnp.average
  elif(kwargs['reduction'] == 'mean'):
    reduction = jnp.mean
  elif(kwargs['reduction'] == 'min'):
    reduction = jnp.min
  elif(kwargs['reduction'] == 'sum'):
    reduction = jnp.sum

  #activation
  activation = jax.nn.relu #default
  
  if(kwargs['activation'] == 'elu'):
    activation = jax.nn.elu
  elif(kwargs['activation'] == 'leaky_relu'):
    activation = jax.nn.leaky_relu
  elif(kwargs['activation'] == 'glu'):
    activation = jax.nn.glu
  elif(kwargs['activation'] == 'sigmoid'):
    activation = jax.nn.sigmoid
  elif(kwargs['activation'] == 'log_sigmoid'):
    activation = jax.nn.log_sigmoid
  elif(kwargs['activation'] == 'hard_sigmoid'):
    activation = jax.nn.hard_sigmoid
  elif(kwargs['activation'] == 'sparse_sigmoid'):
    activation = jax.nn.sparse_sigmoid
  elif(kwargs['activation'] == 'hard_tanh'):  
    activation = jax.nn.hard_tanh

  #gated activation
  gated = kwargs['gated']
  gated_activation = jax.nn.sigmoid #default

  if(kwargs['gated_activation'] == 'hard_sigmoid'):
    gated_activation = jax.nn.hard_sigmoid
  elif(kwargs['gated_activation'] == 'log_sigmoid'):
    gated_activation = jax.nn.log_sigmoid
  elif(kwargs['gated_activation'] == 'sparse_sigmoid'):
    gated_activation = jax.nn.sparse_sigmoid
  elif(kwargs['gated_activation'] == 'hard_tanh'):  
    gated_activation = jax.nn.hard_tanh
  elif(kwargs['gated_activation'] == 'tanh'):  
    gated_activation = jax.nn.tanh
  elif(kwargs['gated_activation'] == 'relu'):
    gated_activation = jax.nn.relu
  elif(kwargs['gated_activation'] == 'elu'):
    gated_activation = jax.nn.elu

  memory_size = kwargs.get('memory_size', None)
  memory_type = kwargs.get('memory_type', None)

  #factory with methods
  def _factory(out_size: int):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln,
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=0,
      )
    elif kind == 'triplet_mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'triplet_pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
      )
    elif kind == 'gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=False,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gpgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif kind == 'triplet_gmpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          gated=True,
      )
    elif 'rt' in kind:
      processor = RT(
          nb_layers=kwargs['nb_layers'],
          nb_heads=nb_heads,
          vec_size=out_size,
          node_hid_size=kwargs['node_hid_size'],
          edge_hid_size_1=kwargs['edge_hid_size_1'],
          edge_hid_size_2=kwargs['edge_hid_size_2'],
          graph_vec=kwargs['graph_vec'],
          disable_edge_updates=kwargs['disable_edge_updates'],
          name=kind
      )
    elif kind == 'f1':
      processor = F1(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f2':
      processor = F2(
          out_size=out_size,
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f3':
      processor = F3(
          out_size=out_size,
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f4':
      processor = F4(
          out_size=out_size,
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f5':
      processor = F5(
          out_size=out_size,
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f6':
      processor = F6(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f7':
      processor = F7(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation
      )
    elif kind == 'f8':
      processor = F8(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln,
          use_triplets=True,
          nb_triplet_fts=nb_triplet_fts,
          activation = activation,
          reduction = reduction,
          gated = gated,
          gated_activation = gated_activation,
          memory_type = memory_type,
          memory_size = memory_size
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)
