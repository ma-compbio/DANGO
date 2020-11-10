import torch
from Modules import *
from sklearn.decomposition import PCA

# Generating list of neighbors for GNNs
def generate_neighbor_list(input, gene_num):
	input = input[:, :-1].astype('int')
	neighbor_list = [[] for i in range(gene_num)]
	
	for datum in input:
		neighbor_list[datum[0]].append(datum[1])
		neighbor_list[datum[1]].append(datum[0])
	
	neighbor_list = [list(set(nb)) + [i] for i, nb in enumerate(neighbor_list)]
	return np.array(neighbor_list)


def generate_neighbor_list_with_weight(input, gene_num):
	# input = input[:, :-1].astype('int')
	neighbor_list = [[] for i in range(gene_num)]
	
	for datum in input:
		neighbor_list[int(datum[0])].append([int(datum[1]), datum[-1]])
		neighbor_list[int(datum[1])].append([int(datum[0]), datum[-1]])
	new_neighbor_list = []
	for nb in neighbor_list:
		new_neighbor_list.append(np.array(nb))
	return np.array(new_neighbor_list)



# get the DangoModel
def get_model(gene_num, embed_dim, auxi_m, auxi_adj):
	first_embedding = torch.nn.Embedding(gene_num, embed_dim, padding_idx=0).to(device)
	
	
	print("embedding size", gene_num, embed_dim)
	
	graphsage_embedding = [GraphEncoder(features=first_embedding, feature_dim=embed_dim,
	                                                embed_dim=embed_dim,
	                                                neighbor_list=generate_neighbor_list(auxi_m[i], gene_num),
	                                                num_sample=5, gcn=False).to(device) for i in range(len(auxi_m))]
	graphsage_embedding = [GraphEncoder(features=graphsage_embedding[i], feature_dim=embed_dim,
	                                    embed_dim=embed_dim, neighbor_list=generate_neighbor_list(auxi_m[i], gene_num),
	                                    num_sample=5, gcn=False).to(device) for i in range(len(auxi_m))]
	recon_nn = [FeedForward([embed_dim, embed_dim, adj.shape[-1]]).to(device) for adj in auxi_adj]
	
	node_embedding = MetaEmbedding(embed_list=graphsage_embedding,
	                               dim=embed_dim).to(device)
	
	hypersagnn = Hyper_SAGNN(
		n_head=8,
		d_model=embed_dim,
		d_k=embed_dim,
		d_v=embed_dim,
		node_embedding=node_embedding,
		diag_mask=True,
		bottle_neck=embed_dim).to(device)
	return graphsage_embedding, recon_nn, node_embedding, hypersagnn



# Get the baselien model
def get_baseline_model(gene_num, embed_dim, auxi_m, auxi_adj):
	first_embedding = torch.nn.Embedding(gene_num, embed_dim, padding_idx=0).to(device)
	print("embedding size", gene_num, embed_dim)
	graphsage_embedding = [GraphEncoder(features=first_embedding, feature_dim=embed_dim,
	                                                embed_dim=embed_dim,
	                                                neighbor_list=generate_neighbor_list(auxi_m[i],
	                                                                                                 gene_num),
	                                                num_sample=5, gcn=True).to(device) for i in range(len(auxi_m))]


	recon_nn = [FeedForward([embed_dim, embed_dim, adj.shape[-1]]).to(device) for adj in auxi_adj]
	node_embedding = MetaEmbedding_Avg(embed_list=graphsage_embedding,
	                               dim=embed_dim).to(device)
	
	hypersagnn = average_MLP(embed_dim, 1, embed = node_embedding).to(device)
	return graphsage_embedding, recon_nn, node_embedding, hypersagnn
