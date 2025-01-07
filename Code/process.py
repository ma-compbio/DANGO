import pandas as pd
import numpy as np
from tqdm import trange, tqdm


# Build the dictionaries that maps gene name to its node id, and the id2genename
# The id starts at 1, 0 is preserved as padding index
def build_dict():
	genenamelist = np.loadtxt("../data/yeast_network/yeast_string_genes.txt", dtype='str')
	gene2id = {}
	id2gene = {}
	
	# We are enforcing the gene id to start with 1, 0 is preserved as padding
	for i, g in enumerate(genenamelist):
		gene2id[g] = i + 1
		id2gene[i + 1] = g
	np.save("../data/gene2id.npy", gene2id)
	np.save("../data/id2gene.npy", id2gene)

# Get the mashup vectors
def get_vectors():
	vecs = np.loadtxt("../data/string_yeast_mashup_vectors_d500.txt", delimiter="\t")
	np.save("../data/embeddings.npy", vecs)


# Generate pairwise networks
def generate_sub_pairwise_network():
	# These files already starts with 1
	for f in ['yeast_string_coexpression_adjacency.txt', 'yeast_string_experimental_adjacency.txt',
	          'yeast_string_database_adjacency.txt', 'yeast_string_neighborhood_adjacency.txt',
	          'yeast_string_fusion_adjacency.txt', 'yeast_string_cooccurence_adjacency.txt','yeast_genes_baker_adjacency.txt']:
		adj = np.loadtxt("../data/yeast_network/%s" % f, dtype='float32')
		file_name = f.split("_")[2]
		
		np.save("../data/%s.npy" % file_name, adj)


# Parsing hyperedges from experimental datasets
def generate_tuple_list():
	tab = pd.read_table("../data/AdditionalDataS1.tsv", sep="\t")
	print(tab)
	group1 = np.array(tab["Query strain ID"])
	group2 = np.array(tab["Array strain ID"])
	score = np.array(tab["Adjusted genetic interaction score (epsilon or tau)"])
	significance = np.array(tab['P-value'])
	genename2id = np.load("../data/gene2id.npy", allow_pickle=True).item()
	tuple_list = []
	label = []
	tuple_sign = []
	
	
	pairwise_list = []
	pairwise_label = []
	pairwise_sign = []
	
	
	unique_gene_list = set()
	trigenic_count = 0
	for i in trange(len(group1)):
		g1 = group1[i]
		g2 = group2[i]
		s = score[i]
		sg = significance[i]
		
		# g1 actually only has one mutation
		if "YDL227C" in g1:
			g1 = str(g1).split("_")[0]
			g11, g12 = g1.split("+")

			
			if 'YDL227C' in g11:
				g1 = g12
			elif 'YDL227C' in g12:
				g1 = g11
			else:
				print ("error", g1, g11, g12)
				
			g2 = str(g2).split("_")[0]
			if (g1 in genename2id) and (g2 in genename2id):
				temp = [genename2id[g1], genename2id[g2]]
				temp.sort()
				pairwise_list.append(temp)
				pairwise_label.append(s)
				pairwise_sign.append(sg)
		elif "YDL227C" in g2:
			g1 = str(g1).split("_")[0]
			g11, g12 = g1.split("+")
			if (g11 in genename2id) and (g12 in genename2id):
				temp = [genename2id[g11], genename2id[g12]]
				temp.sort()
				pairwise_list.append(temp)
				pairwise_label.append(s)
				pairwise_sign.append(sg)
		else:
			trigenic_count += 1
			g1 = str(g1).split("_")[0]
			try:
				g11, g12 = g1.split("+")
			except:
				print (i, g1)
				raise EOFError
			
			
		
					
			
			g2 = str(g2).split("_")[0]
			
			for g in [g11, g12, g2]:
				if g not in unique_gene_list:
					unique_gene_list.add(g)
					
					
			if (g11 in genename2id) and (g12 in genename2id) and (g2 in genename2id):
				temp = [genename2id[g11], genename2id[g12], genename2id[g2]]
				temp.sort()
				tuple_list.append(temp)
				label.append(s)
				tuple_sign.append(sg)
			
	tuple_list = np.array(tuple_list)
	label = np.array(label)
	
	pairwise_list = np.array(pairwise_list)
	pairwise_label = np.array(pairwise_label)
	print ()
	print(tuple_list.max())
	print ("trigenic_count", trigenic_count)
	print ("unique_genes", len(unique_gene_list), np.unique(list(unique_gene_list)).shape)
	print ("mapped genes", np.unique(tuple_list.reshape((-1))).shape)
	print (tuple_list.shape, np.unique(tuple_list,axis=0).shape)
	np.save("../data/tuples.npy", tuple_list)
	np.save("../data/y.npy", label)
	np.save("../data/sign.npy", tuple_sign)
	
	np.save("../data/pairs.npy", pairwise_list)
	np.save("../data/pair_y.npy", pairwise_label)
	np.save("../data/pair_sign.npy", pairwise_sign)
	
	print (pairwise_list.shape)



	
	
		
build_dict()
get_vectors()
generate_sub_pairwise_network()
generate_tuple_list()