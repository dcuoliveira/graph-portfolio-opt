import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Batch


Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Batches = Union[np.ndarray, None]
Additional_Features = Sequence[np.ndarray]


class ForecastBatch(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Batch object. Between two
    temporal snapshots the feature matrix, target matrices and optionally passed
    attributes might change. However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        batches (Numpy array): Batch index tensor.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        window_length: int,
        step_length: int,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.window_length = window_length
        self.step_length = step_length
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()
        
    def get_num_batches(self) -> int:
        if (self.features.shape[2] - self.window_length) % self.step_length == 0:
            return int((self.features.shape[2] - self.window_length) / self.step_length)
        else:
            return int((self.features.shape[2] - self.window_length) / self.step_length) + 1

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_feature(self, time_index: int):
        start_ind = time_index * self.step_length
        end_ind  = start_ind + self.window_length
        
        return torch.unsqueeze(torch.FloatTensor(self.features[:, :, start_ind:end_ind]), 0)

    def _get_target(self, time_index: int):
        start_ind = time_index * self.step_length
        end_ind  = start_ind + self.window_length + 1
        return torch.FloatTensor(self.targets[:, start_ind:end_ind])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: int):
        x = self._get_feature(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
        y = self._get_target(time_index)
        additional_features = self._get_additional_features(time_index)
        snapshot = Batch(x=x, edge_index=edge_index, edge_attr=edge_weight,
                        y=y, **additional_features)
        return snapshot

    def __next__(self):
        if (self.t * self.step_length) + self.window_length < self.features.shape[2]:
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self