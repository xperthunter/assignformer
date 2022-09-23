import torch
from torch import nn


class PeakAttention(nn.Module)
	class __init__(
		self,
		dim,
		dim_head,
		heads
	):
	super().__init__()
	
	self.norm = nn.LayerNorm(dim)
	
	self.attn = Attention()
	
	def forward



class AssignformerBlock(nn.Module):
	def __init__(
		self,
		dim,
		dim_head,
		heads
	):
	super().__init__()
	
	self.peak_attn           = PeakAttention()
	self.seq_attn            = SeqAttention()
	self.peak_seq_cross_attn = PeakSeqCross()
	
	def forward(
		self,
		x,
		mask = None
	):
		
		x = self.peak_attn(x, mask = mask) + x
		x = self.seq_attn(x, mask = mask) + x
		x = self.peak_seq_cross_attn(x, mask = mask)
		
		return x
		

class Trunk(nn.Module):
	def __init__(
		self,
		dim,
		dim_head,
		depth,
		heads):
		"""
		* dim	Embedding dimension
		* dim_head	Size of queries, keys, and values
		* depth	Number of layers
		* heads	Number of heads
		"""
		super().__init__()
		
		self.layers = nn.ModuleList([
			AssignFormerBlock(
				dim,
				dim_head,
				heads
			) for _ in range(depth)])


class assignformer(nn.Module):
	def __init__(
		self,
		peak_dim = 4,
		seq_dim = 8,
		dim = 16,
		dim_head = 32,
		depth = 4,
		heads = 4,
		):
		"""
		* peak_dim Input dimension of peak info (4 == HNCCA)
		* seq_dim Input dimension of sequence features (8 == 2xpeak_info)
		* dim	Embedding dimension
		* dim_head Dimesion for creation of queries, keys, values
		* depth Depth of Transformer
		* heads Number of Attention Heads
		* 
		"""
		super().__init__()
		self.peak_dim = peak_dim
		self.seq_dim  = seq_dim
		self.dim = dim
		
		self.peak_embedding = nn.Linear(self.peak_dim, dim)
		self.seq_embedding  = nn.Linear(self.seq_dim, dim)
		
		self.trunk = Trunk(
			dim,
			dim_head,
			depth,
			heads
		)
		
		