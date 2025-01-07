import numpy as np
import h5py
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

file = h5py.File("../data/string_prose_embeddings.h5")
ppi_embeddings = dict()
embedding_size = 0
for key, val in file.items():
    ppi_embeddings[key[5:]] = val[:]

# print(ppi_embeddings, len(ppi_embeddings), len(ppi_embeddings["YPR204W"]))

genes = np.loadtxt("../data/yeast_network/yeast_string_genes.txt", dtype=str)
ppi_embeddings_list = []
notin = 0
isin = 0
rightlength = set()
for i, gene in enumerate(genes):
    if gene in ppi_embeddings:
        isin += 1
    if gene not in ppi_embeddings:
        notin += 1
        ppi_embeddings[gene] = np.zeros(6165).tolist()
    rightlength.add(len(ppi_embeddings[gene]))
    ppi_embeddings_list.append(ppi_embeddings[gene])
ppi_embeddings_list.insert(0, np.zeros(6165).tolist())
print(rightlength)
ppi_embeddings_list = np.array(ppi_embeddings_list)
print(notin,isin, notin+isin)
print(ppi_embeddings_list.shape)
print(ppi_embeddings_list[0])
print(ppi_embeddings_list)
np.save("../data/ppi_embeddings", ppi_embeddings_list)


embeddings = np.load("../data/ppi_embeddings.npy")
# print(np.zeros(shape=(6165,)), embeddings.shape, embeddings[0].shape == np.zeros(shape=(6165,)).shape)

# embeddings = np.insert(embeddings, 0, np.zeros(shape=(6165,1)))
print(embeddings.shape)
print(embeddings[0].shape)
print(embeddings)
print(np.zeros(6165))
embeddings = np.concatenate((np.zeros((6165,1)).T, embeddings), axis=1)
print(embeddings.shape)
print(embeddings[0].shape)

gene2id = np.load("../data/gene2id.npy", allow_pickle=True)
print(gene2id)

# id2gene = np.load("../data/id2gene.npy", allow_pickle=True)
# print(id2gene)