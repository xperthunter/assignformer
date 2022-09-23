import sys

import torch.utils

class AssignmentDataset(torch.utils.data.Dataset):
	""" 
	This is the dataset that holds lists of peak lists and sequence features 
	"""
	
	def __init__(
		self,
		dataframe
	):
		
		self.peaks = dataframe['peaks']
		self.seqs  = dataframe['seqs']
		
		self.peaks_masks = dataframe['peaks_masks']
		self.