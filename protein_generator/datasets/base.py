import torch

import dgl
from dgl.data import DGLDataset


class DataModule(DGLDataset):
	
	def __getitem__(self, i):
		pass

	def __len__(self):
		pass

	def process(self):
		pass

	def _collate(self, graphs, node_features, edge_features):
		"""
		A collate function which allows you to batch multiple samples from the dataset
		"""
		batched_graph = dgl.batch(graphs)
		node_features = self.combine_features(node_features)
		edge_features = self.combine_features(edge_features)
		return batched_graph, node_features, edge_features

	@staticmethod
	def combine_features(features):
		"""
		Takes a dictionary with l0 and l1 features and combines the features for batch processing
		:param features: a list of dictionaries which contain "feature level (eg 0, 1)": a torch Tensor containing features
		:return: a dictionary mapping "feature level": torch Tensor with the features for all the batched sample features concatenated
		"""
		combined_features = {}
		for feat in features:
			for key, val in feat.items:
				if key not in combined_features:
					combined_features[key] = []
				combined_features[key].append(val)
		for key, val in combined_features.items:
			combined_features[key] = torch.cat(*val)		
		return combined_features
