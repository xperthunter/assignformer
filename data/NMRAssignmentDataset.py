import torch.utils

class NMRAssignmentDataset(torch.utils.data.Dataset):
	""" 
	This is the dataset that holds lists of peak lists and sequence features 
	"""
	
	def __init__(
		self,
		dataframe):
		
		self.names = dataframe['bmrbids']
		self.peaks = dataframe['peaks']
		self.seqs  = dataframe['seqs']
		
		self.peaks_masks = dataframe['peaks_masks']
		self.seqs_masks = dataframe['seqs_masks']
	
	def __len__(self):
		return len(self.peaks)
	
	def __getitem__(self, idx):
		return (
			self.names[idx],
			self.peaks[idx],
			self.peaks_masks[idx],
			self.seqs[idx],
			self.seqs_masks[idx]
		)
	
	def __str__(self):
		return 'peaks and seqs dataset'
	
	def __repr__(self):
		return self.__str__()
		
		