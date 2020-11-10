import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math
from utils import *
import multiprocessing
from torch.nn.utils.rnn import pad_sequence

cpu_num = multiprocessing.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activation_func = torch.tanh

def swish(x):
	return x * torch.sigmoid(x)

activation_func = swish



def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
	''' For masking out the padding part of key sequence. '''
	
	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(0)
	padding_mask = padding_mask.unsqueeze(
		1).expand(-1, len_q, -1)  # b x lq x lk
	
	return padding_mask



class MetaEmbedding_Avg(nn.Module):
	def __init__(self, embed_list, dim):
		super().__init__()
		self.embed_list = embed_list
		self.dim = dim
		
		
		for i in range(len(self.embed_list)):
			self.add_module("embed_nn_%d" % i, self.embed_list[i])
	
	def forward(self, x):
		if len(x.shape) > 1:
			sz_b, len_seq = x.shape
			x = x.view(-1)
			reshape_flag = True
		else:
			reshape_flag = False
		
		
		embed = torch.stack([embed_nn(x) for embed_nn in self.embed_list], dim=-1)
		# shape of (sz_b * len_seq, n_embed, dim)
		embed = embed.permute(0, 2, 1)
		# shape of (sz_b * len_seq, n_embed, 1)
		
		embed = torch.mean(embed, dim=-2, keepdim=False)
		
		if reshape_flag:
			embed = embed.view(sz_b, len_seq, -1)
		#
		return embed


class MetaEmbedding(nn.Module):
	def __init__(self, embed_list, dim):
		super().__init__()
		self.embed_list = nn.ModuleList(embed_list)
		self.dim = dim
		self.attention = nn.Linear(dim, 1)
		
		self.forward = self.on_hook_forward
		
	# Can be used during test for acceleration
	def off_hook(self, ids, size):
		with torch.no_grad():
			if type(ids) == np.ndarray:
				ids = torch.LongTensor(ids).to(device)
				embeds = self.forward(ids)
				self.off_hook_embedding = torch.zeros(size, embeds.shape[-1]).float().to(device)
				self.off_hook_embedding[ids] = embeds
				self.forward = self.off_hook_forward
	
	def on_hook(self):
		del self.off_hook_embedding
		self.forward = self.on_hook_forward
	
	def off_hook_forward(self, x):
		if len(x.shape) > 1:
			sz_b, len_seq = x.shape
			x = x.view(-1)
			reshape_flag = True
		else:
			reshape_flag = False
			
		embed =  self.off_hook_embedding[x]
		if reshape_flag:
			embed = embed.view(sz_b, len_seq,-1)
		#
		return embed
		
	def on_hook_forward(self, x):
		if len(x.shape) > 1:
			sz_b, len_seq = x.shape
			x = x.view(-1)
			reshape_flag = True
		else:
			reshape_flag = False
		
		
		embed = torch.stack([embed_nn(x) for embed_nn in self.embed_list], dim=-1)
		# shape of (sz_b * len_seq, n_embed, dim)
		embed = embed.permute(0, 2, 1)
		# shape of (sz_b * len_seq, n_embed, 1)

		weight = self.attention(embed)
		weight = F.softmax(weight, dim=-2)
		embed = torch.sum(embed * weight, dim=-2, keepdim=False)
		
		if reshape_flag:
			embed = embed.view(sz_b, len_seq,-1)
			
		return embed


class Hyper_SAGNN(nn.Module):
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			node_embedding,
			diag_mask,
			bottle_neck):
		super().__init__()
		
		self.pff_classifier = PositionwiseFeedForward(
			[d_model, 1])
		
		self.node_embedding = node_embedding
		
		self.encode1 = EncoderLayer(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul=0.3,
			dropout_pff=0.4,
			diag_mask=diag_mask,
			bottle_neck=bottle_neck)
		self.encode2 = EncoderLayer(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul=0.1,
			dropout_pff=0.1,
			diag_mask=diag_mask,
			bottle_neck=d_model)
		self.diag_mask_flag = diag_mask
		
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(0.25)
		
		self.add_module("graph_ndoe_embeddings", self.node_embedding)
	
	def get_embedding(self, x, slf_attn_mask=None, non_pad_mask=None):
		if slf_attn_mask is None:
			slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
			non_pad_mask = get_non_pad_mask(x)
		
		x = self.node_embedding(x)
		dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
		# dynamic, static, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
		return dynamic, static, attn
	
	def forward(self, x, mask=None):
		x = x.long()
		slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
		non_pad_mask = get_non_pad_mask(x)
		
		dynamic, static, attn = self.get_embedding(x, slf_attn_mask, non_pad_mask)
		dynamic = self.layer_norm1(dynamic)
		static = self.layer_norm2(static)
		
		if self.diag_mask_flag == 'True':
			output = (dynamic - static) ** 2
		else:
			output = dynamic
		
		output = self.dropout(output)
		output = self.pff_classifier(output)
		
		mode = 'sum'
		
		output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
			
		return output
	
	def predict(self, input, verbose=False, batch_size=960):
		self.eval()
		with torch.no_grad():
			output = []
			if verbose:
				func1 = trange
			else:
				func1 = range
			if batch_size < 0:
				batch_size = len(input)
			with torch.no_grad():
				for j in func1(math.ceil(len(input) / batch_size)):
					x = input[j * batch_size:min((j + 1) * batch_size, len(input))]
					output.append(self(x))
			output = torch.cat(output, dim=0)
			torch.cuda.empty_cache()
		self.train()
		return output


# A custom position-wise MLP.
# dims is a list, it would create multiple layer with tanh between them
# If dropout, it would add the dropout at the end. Before residual and
# layer-norm
class PositionwiseFeedForward(nn.Module):
	def __init__(
			self,
			dims,
			dropout=None,
			reshape=False,
			use_bias=True,
			residual=False,
			layer_norm=False):
		super(PositionwiseFeedForward, self).__init__()
		self.w_stack = []
		self.dims = dims
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, use_bias))
			self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
		self.reshape = reshape
		self.layer_norm = nn.LayerNorm(dims[0])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.residual = residual
		self.layer_norm_flag = layer_norm
		self.alpha = torch.nn.Parameter(torch.zeros(1))
		
		self.register_parameter("alpha", self.alpha)
	
	def forward(self, x):
		if self.layer_norm_flag:
			output = self.layer_norm(x)
		else:
			output = x
		output = output.transpose(1, 2)
		
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation_func(output)
			if self.dropout is not None:
				output = self.dropout(output)
		
		output = self.w_stack[-1](output)
		output = output.transpose(1, 2)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		if self.dims[0] == self.dims[-1]:
			# residual
			if self.residual:
				output = self.alpha * output + x
		
		return output


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier
class FeedForward(nn.Module):
	''' A two-feed-forward-layer module '''
	
	def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
		super(FeedForward, self).__init__()
		self.w_stack = []
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
			self.add_module("FF_Linear%d" % (i), self.w_stack[-1])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.reshape = reshape
	
	def forward(self, x):
		output = x
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation_func(output)
		if self.dropout is not None:
			output = self.dropout(output)
		output = self.w_stack[-1](output)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		return output
	
	def fit(self, X, y, epochs=10, early_stop=True, verbose=True, projection_matrix=None, stop_loss=None):
		# for name, param in self.named_parameters():
		# 	print(name, param.requires_grad, param.shape)
		print (X.shape)
		if projection_matrix is not None:
			print (torch.sum(projection_matrix), projection_matrix.shape)
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		data = torch.from_numpy(X).to(device).float()
		y = torch.from_numpy(y).to(device).float()
		
		if verbose:
			bar = trange(epochs, desc="")
		else:
			bar = range(epochs)
			
		loss_best = 1e5
		no_improve_count = 0
		if projection_matrix is not None:
			# print (self.w_stack[0].weight.data.shape, projection_matrix.shape)
			self.w_stack[0].weight.data = self.w_stack[0].weight.data * projection_matrix
			
		for i in bar:
			pred = self.forward(data)
			optimizer.zero_grad()
			loss = F.mse_loss(pred, y, reduction="sum") / len(pred)
			
			loss.backward()
			optimizer.step()
			
			if projection_matrix is not None:
				# print (self.w_stack[0].weight.data.shape, projection_matrix.shape)
				self.w_stack[0].weight.data = self.w_stack[0].weight.data * projection_matrix
			
			if stop_loss is not None and loss.item() <= stop_loss:
				break
			if i >= 20:
				if loss.item() < loss_best:
					loss_best = loss.item()
					no_improve_count = 0
				else:
					no_improve_count += 1
				
				
			if early_stop:
				if no_improve_count >= 100:
					break
			
			bar.set_description("%.3f" % (loss.item()), refresh=False)
		
		print("loss", loss.item(), "loss best", loss_best, "epochs", i)
		print()
		torch.cuda.empty_cache()
	
	def predict(self, data):
		self.eval()
		data = torch.from_numpy(data).to(device).float()
		with torch.no_grad():
			encode = self.forward(data)
		self.train()
		torch.cuda.empty_cache()
		return encode.cpu().detach().numpy()


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier

class ScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product Attention '''
	
	def __init__(self, temperature):
		super().__init__()
		self.temperature = temperature
	
	def masked_softmax(self, vector: torch.Tensor,
	                   mask: torch.Tensor,
	                   dim: int = -1,
	                   memory_efficient: bool = False,
	                   mask_fill_value: float = -1e32) -> torch.Tensor:
		
		if mask is None:
			result = torch.nn.functional.softmax(vector, dim=dim)
		else:
			mask = mask.float()
			while mask.dim() < vector.dim():
				mask = mask.unsqueeze(1)
			if not memory_efficient:
				# To limit numerical errors from large vector elements outside
				# the mask, we zero these out.
				result = torch.nn.functional.softmax(vector * mask, dim=dim)
				result = result * mask
				result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
			else:
				masked_vector = vector.masked_fill(
					(1 - mask).bool(), mask_fill_value)
				result = torch.nn.functional.softmax(masked_vector, dim=dim)
		return result
	
	def forward(self, q, k, v, diag_mask, mask=None):
		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature
		
		if mask is not None:
			attn = attn.masked_fill(mask, -float('inf'))
		
		attn = self.masked_softmax(
			attn, diag_mask, dim=-1, memory_efficient=True)
		output = torch.bmm(attn, v)
		
		return output, attn


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout,
			diag_mask,
			input_dim):
		super().__init__()
		
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
		
		nn.init.normal_(self.w_qs.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_v)))
		
		self.attention = ScaledDotProductAttention(
			temperature=np.power(d_k, 0.5))
		
		self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
		self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
		
		self.layer_norm1 = nn.LayerNorm(input_dim)
		self.layer_norm2 = nn.LayerNorm(input_dim)
		self.layer_norm3 = nn.LayerNorm(input_dim)
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = dropout
		
		self.diag_mask_flag = diag_mask
		self.diag_mask = None
		
		self.alpha_static = torch.nn.Parameter(torch.zeros(1))
		self.alpha_dynamic = torch.nn.Parameter(torch.zeros(1))
		
		self.register_parameter("alpha_static", self.alpha_static)
		self.register_parameter("alpha_dynamic", self.alpha_dynamic)
	
	def forward(self, q, k, v, diag_mask, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		
		residual_dynamic = q
		residual_static = v
		
		q = self.layer_norm1(q)
		k = self.layer_norm2(k)
		v = self.layer_norm3(v)
		
		sz_b, len_q, _ = q.shape
		sz_b, len_k, _ = k.shape
		sz_b, len_v, _ = v.shape
		
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		
		q = q.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_q, d_k)  # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_k, d_k)  # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_v, d_v)  # (n*b) x lv x dv
		
		n = sz_b * n_head
		if self.diag_mask is not None:
			if (len(self.diag_mask) <= n) or (
					self.diag_mask.shape[1] != len_v):
				self.diag_mask = torch.ones((len_v, len_v), device=device)
				if self.diag_mask_flag == 'True':
					self.diag_mask -= torch.eye(len_v, len_v, device=device)
				self.diag_mask = self.diag_mask.repeat(n, 1, 1).bool()
				diag_mask = self.diag_mask
			else:
				diag_mask = self.diag_mask[:n]
		
		else:
			self.diag_mask = (torch.ones((len_v, len_v), device=device))
			if self.diag_mask_flag == 'True':
				self.diag_mask -= torch.eye(len_v, len_v, device=device)
			self.diag_mask = self.diag_mask.repeat(n, 1, 1).bool()
			diag_mask = self.diag_mask
		
		if mask is not None:
			mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
		
		dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
		
		dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
		dynamic = dynamic.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		static = v.view(n_head, sz_b, len_q, d_v)
		static = static.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		
		dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
		static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
		
		dynamic = dynamic #* self.alpha_dynamic + residual_dynamic
		
		static = static #* self.alpha_static + residual_static
		
		return dynamic, static, attn


class EncoderLayer(nn.Module):
	'''A self-attention layer + 2 layered pff'''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul,
			dropout_pff,
			diag_mask,
			bottle_neck):
		super().__init__()
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.mul_head_attn = MultiHeadAttention(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout=dropout_mul,
			diag_mask=diag_mask,
			input_dim=bottle_neck)
		self.pff_n1 = PositionwiseFeedForward(
			[d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
		
		residual = True if bottle_neck == d_model else False
		self.pff_n2 = PositionwiseFeedForward(
			[bottle_neck, d_model], dropout=dropout_pff, residual=residual, layer_norm=True)
	
	# self.dropout = nn.Dropout(0.2)
	
	def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
		dynamic, static1, attn = self.mul_head_attn(
			dynamic, dynamic, static, slf_attn_mask)
		dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
		# static = self.pff_n2(static * non_pad_mask) * non_pad_mask
		
		return dynamic, static1, attn
	
class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	
	def __init__(self, features, init_transform=None, gcn=False):
		"""
		Initializes the aggregator for a specific graph.
		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""
		
		super(MeanAggregator, self).__init__()
		
		self.features = features
		self.gcn = gcn
		self.init_transform = init_transform
	
	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
			                            num_sample,
			                            )) if len(to_neigh) >= num_sample else _set(to_neigh) for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs
		
		if self.gcn:
			nodes = nodes.cpu()
			samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]
		
		
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = torch.zeros(len(samp_neighs), len(unique_nodes), device=device)
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		mask = mask.to(device)
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh + 1e-15 + int(self.gcn))
		embed_matrix = self.init_transform(self.features(torch.LongTensor(unique_nodes_list).to(device)))
		to_feats = mask.mm(embed_matrix)
		return to_feats
	

# GraphSAGE GNN
class GraphEncoder(nn.Module):
	def __init__(self, features, feature_dim,
			embed_dim, neighbor_list,
			num_sample=10, gcn=False):
		super(GraphEncoder, self).__init__()
		self.features = features
		self.feature_dim = feature_dim
		self.embed_dim = embed_dim
		self.neighbor_list = neighbor_list
		self.num_sample=num_sample
		self.gcn=gcn
		self.init_transform = nn.Linear(feature_dim, embed_dim)
		input_size = 1
		if not self.gcn:
			input_size += 1
		self.final_transform = nn.Linear(input_size * self.feature_dim, embed_dim)
		self.aggregator = MeanAggregator(self.features, self.init_transform, self.gcn)
		self.forward = self.on_hook_forward
	
	def on_hook_forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.
		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.neighbor_list[int(node)] for node in nodes.cpu()],
		                                      self.num_sample)
		if not self.gcn:
			self_feats = self.init_transform(self.features(nodes))
			combined = torch.cat([self_feats, neigh_feats], dim=1)
		else:
			combined = neigh_feats
		combined = activation_func(self.final_transform(combined))
		return combined
	
	def off_hook(self, ids, size):
		with torch.no_grad():
			if type(ids) == np.ndarray:
				ids = torch.LongTensor(ids).to(device)
				embeds = self.forward(ids)
				self.off_hook_embedding = torch.zeros(size, embeds.shape[-1]).float().to(device)
				self.off_hook_embedding[ids] = embeds
				self.forward = self.off_hook_forward
	
	def on_hook(self):
		del self.off_hook_embedding
		self.forward = self.on_hook_forward
	
	def off_hook_forward(self, nodes):
		
		return self.off_hook_embedding[nodes.view(-1)]


class MeanAggregator_with_weights(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	
	def __init__(self, features, init_transform=None, gcn=False, dim=0):
		"""
		Initializes the aggregator for a specific graph.
		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""
		
		super(MeanAggregator_with_weights, self).__init__()
		
		self.features = features
		self.gcn = gcn
		self.init_transform = init_transform
		self.dim = dim
	
	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)

		samp_neighs = np.asarray(to_neighs)
		# sample_if_long_enough = lambda to_neigh: to_neigh[np.random.permutation(to_neigh.shape[0])[:num_sample]] if len(
		# 	to_neigh) > num_sample else to_neigh
		# if self.training:
		# 	# samp_neighs = np.array([np.array(sample_if_long_enough(to_neigh)) for to_neigh in to_neighs])
		# 	samp_neighs = []
		# 	for to_neigh in to_neighs:
		# 		samp_neighs.append(np.array(sample_if_long_enough(to_neigh)))
		# 	# print (samp_neighs)
		# 	# samp_neighs = np.array(samp_neighs, dtype='object')
		# else:
		# 	samp_neighs = np.asarray(to_neighs)
		unique_nodes = {}
		unique_nodes_list = []
		
		count = 0
		column_indices = []
		row_indices = []
		v = []
		
		for i, samp_neigh in enumerate(samp_neighs):
			if len(samp_neigh) == 0:
				continue
			w = samp_neigh[:, 1]
			samp_neigh = samp_neigh[:, 0].astype('int')
			w /= np.sum(w)
			
			for n in samp_neigh:
				if n not in unique_nodes:
					unique_nodes[n] = count
					unique_nodes_list.append(n)
					count += 1
					
				column_indices.append(unique_nodes[n])
				
				row_indices.append(i)
			v.append(w)
		if len(v) == 0:
			#print (nodes, samp_neighs)
			return torch.zeros((len(nodes), self.dim)).float().to(device)
		v = np.concatenate(v, axis=0)
		
	
		
		unique_nodes_list = torch.LongTensor(unique_nodes_list)
		unique_nodes_list = unique_nodes_list.to(device)
		
		
		mask = torch.sparse.FloatTensor(torch.LongTensor([row_indices, column_indices]),
		                                torch.tensor(v, dtype=torch.float),
		                                torch.Size([len(samp_neighs), len(unique_nodes_list)])).to(device)
		embed_matrix = self.features(unique_nodes_list)
		to_feats = mask.mm(embed_matrix)
		return to_feats
		
		# if self.gcn:
		# 	nodes = nodes.cpu()
		# 	samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]
		#
		# unique_nodes_list = list(set.union(*samp_neighs))
		# unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		# mask = torch.zeros(len(samp_neighs), len(unique_nodes), device=device)
		# column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		# row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		# mask[row_indices, column_indices] = 1
		# mask = mask.to(device)
		# num_neigh = mask.sum(1, keepdim=True)
		# mask = mask.div(num_neigh + 1e-15)
		# embed_matrix = self.init_transform(self.features(torch.LongTensor(unique_nodes_list).to(device)))
		# to_feats = mask.mm(embed_matrix)
		# return to_feats


class GraphEncoder_with_weight(nn.Module):
	def __init__(self, features, feature_dim,
	             embed_dim, neighbor_list,
	             num_sample=10, gcn=False):
		super(GraphEncoder_with_weight, self).__init__()
		self.features = features
		self.feature_dim = feature_dim
		self.embed_dim = embed_dim
		self.neighbor_list = neighbor_list
		self.num_sample = num_sample
		self.gcn = gcn
		self.init_transform = nn.Linear(feature_dim, embed_dim)
		input_size = 1
		if not self.gcn:
			input_size += 1
		self.final_transform = nn.Linear(input_size * self.feature_dim, embed_dim)
		self.aggregator = MeanAggregator_with_weights(self.features, self.init_transform, self.gcn, embed_dim)
		self.off_hook_embedding = None
		self.forward = self.on_hook_forward
		
	def on_hook_forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.
		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, self.neighbor_list[nodes.cpu().numpy()],
		                                      self.num_sample)
		if not self.gcn:
			# print ("at least I'm here")
			self_feats = nodes
			# print("self_feats", self_feats)
			self_feats = self.features(self_feats)
			# print ("self_feats", self_feats)
			self_feats = self.init_transform(self_feats)
			# print("self_feats", self_feats)
			combined = torch.cat([self_feats, neigh_feats], dim=1)
			# print ("combined", combined)
		else:
			combined = neigh_feats
		combined = activation_func(self.final_transform(combined))
		return combined
	
	def off_hook(self, ids, size):
		with torch.no_grad():
			if type(ids) == np.ndarray:
				ids = torch.LongTensor(ids).to(device)
				embeds = self.forward(ids)
				self.off_hook_embedding = torch.zeros(size, embeds.shape[-1]).float().to(device)
				self.off_hook_embedding[ids] = embeds
				self.forward = self.off_hook_forward
	def on_hook(self):
		del self.off_hook_embedding
		self.forward = self.on_hook_forward
	
	def off_hook_forward(self, nodes):
		return self.off_hook_embedding[nodes.view(-1)]