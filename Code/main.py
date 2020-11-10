import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import get_model
from train import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import datetime
import h5py
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import subprocess
import argparse
from itertools import combinations
import random
import shutil
# This function is splitting data to make all test data nodes are unseen in training

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi main program")
	parser.add_argument('-e', '--extra', type=str, default="")
	parser.add_argument('-t', '--thread', type=int, default=8)
	parser.add_argument('-m', '--mode', type=str, default='train')
	parser.add_argument('-i', '--identifier', type=str, default=None)
	parser.add_argument('-s', '--split', type=int, default=0)
	parser.add_argument('-p', '--predict', type=int, default=0)
	return parser.parse_args()



def gene_split_1(data, test_id_list):
	train_index, test_index = [], []
	for i, d in enumerate(data):
		if np.sum(np.isin(d, test_id_list)) > 0:
			test_index.append(i)
		else:
			train_index.append(i)
	return np.array(train_index), np.array(test_index)

def gene_split_2(data, test_id_list):
	train_index, test_index = [], []
	for i, d in enumerate(data):
		if np.sum(np.isin(d, test_id_list)) == 3:
			test_index.append(i)
		elif np.sum(np.isin(d, test_id_list)) > 0:
			continue
		else:
			train_index.append(i)
	return np.array(train_index), np.array(test_index)


def id2baselinevecs(vecs, ids):
	final = 0
	for i in range(ids.shape[-1]):
		final += vecs[ids[:, i] - 1]
	return final


def baseline(train_data, train_y, test_data, test_y):
	

	print ("start baseline")
	baseline_vec = np.load("../data/embeddings.npy")
	if args.random == "True":
		np.random.shuffle(baseline_vec)
	baseline_feature_train = id2baselinevecs(baseline_vec, train_data)
	baseline_feature_test = id2baselinevecs(baseline_vec, test_data)
	

	xgb = XGBRegressor(n_jobs=cpu_num, n_estimators=500,max_depth=10).fit(baseline_feature_train, train_y)
	y_pred = xgb.predict(baseline_feature_test)
	print ("xgboost baseline")
	print (y_pred.shape, test_y.shape)
	print(correlation_cuda(test_y, y_pred))
	pos_mask = test_y >= positive_thres
	print(correlation_cuda(test_y[pos_mask], y_pred[pos_mask]))
	print(roc_auc_cuda(pos_mask, y_pred, balance=True))
	y_pred1 = np.copy(y_pred)
	
	rf = RandomForestRegressor(n_jobs=cpu_num, n_estimators=500,max_depth=10).fit(baseline_feature_train, train_y)
	y_pred = rf.predict(baseline_feature_test)

	print ("rf baseline")
	print (correlation_cuda(test_y, y_pred))
	pos_mask = test_y >= positive_thres
	print (correlation_cuda(test_y[pos_mask], y_pred[pos_mask]))
	print(roc_auc_cuda(pos_mask, y_pred, balance=True))
	return y_pred1, y_pred


# call subprocess for training a Dango instance (used in multiprocessing)
def mp_train(positive_thres, gene_num, split_loc, save_dir, save_string):
	cmd = ["python", "train.py", str(positive_thres), str(gene_num), str(embed_dim), split_loc, save_dir, save_string]
	subprocess.call(cmd)
	print("start reading")



def random_cv_nn_experiments(n_fold=5, rounds=10):
	print("random_cv_params", n_fold, rounds)
	kf = KFold(n_splits=n_fold, shuffle=True)
	f = h5py.File(os.path.join("../Temp/%s/Experiment.hdf5" % datetime_object), "w")
	result_file = open("../Temp/%s/result.txt" % (datetime_object), "a")
	cv_count = 0
	checkpoint_list_all = []
	# Separating train_valid / test with k-fold
	for train_valid_index, test_index in kf.split(tuples):
		# Start multiple training instances
		pool = ProcessPoolExecutor(max_workers=args.thread)
		p_list = []
		checkpoint_list = []
		grp = f.create_group("Cross_valid_%d" % cv_count)
		
		ensemble = 0
		test_y = y[test_index]
		save_string_list = []
		for i in range(rounds):
			index = torch.randperm(len(train_valid_index))
			# Randomly separate train and valid 10 times
			train_index = train_valid_index[index[:int(0.8 * len(index))]]
			valid_index = train_valid_index[index[int(0.8 * len(index)):]]
			
			indexs = np.array([train_index, valid_index, test_index])
			split_loc = "../Temp/%s/Experiment_%d_%d_ind.npy" % (datetime_object, cv_count, i)
			save_string = "%d_%d" % (cv_count, i)
			save_string_list.append(save_string)
			np.save(split_loc, indexs)
			p_list.append(pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object,
			                          save_string))
			time.sleep(60)
		
		finish_count = 0
		
		for save_string in save_string_list:
			print("Now at Fold %d iteration %d" % (cv_count, finish_count))
			# Some of the child process might got killed due to memory/GPU memory usage
			# Or someone you are sharing the machine with decide to run a gigantic process in the middle
			# The try / exception here would skip those killed child process
			# so that you won't get frustrated about running the 5-fold CV from scracth
			# But if most of teh child process got killed... Run the program again...
			try:
				ckpt_list = list(
					torch.load("../Temp/%s/%s_model_list" % (datetime_object, save_string), map_location='cpu'))
				ensemble_part = np.load("../Temp/%s/ensemble_%s.npy" % (datetime_object, save_string))
				finish_count += 1
				checkpoint_list += list(ckpt_list)
				ensemble += ensemble_part
				
			except:
				print("some training instances got killed")
		
		ensemble /= finish_count
		
		
		print("ensemble")
		print(correlation_cuda(test_y, ensemble))
		pos_mask = test_y >= positive_thres
		print(correlation_cuda(test_y[pos_mask], ensemble[pos_mask]))
		print(roc_auc_cuda(pos_mask, ensemble, balance=True))
		
		corr1, corr2 = correlation_cuda(test_y, ensemble)
		pos_mask = test_y >= positive_thres
		corr1_pos, corr2_pos = correlation_cuda(test_y[pos_mask], ensemble[pos_mask])
		roc, aupr = roc_auc_cuda(pos_mask, ensemble, balance=True)
		result_file.write("Fold %d\n" % (cv_count))
		for score in [corr1, corr2, corr1_pos, corr2_pos, roc, aupr]:
			result_file.write("%f\n" % (score))
		sys.stdout.flush()
		
		print("Start saving")
		# grp.create_dataset("train_index", data=train_index)
		# grp.create_dataset("valid_index", data=valid_index)
		# grp.create_dataset("test_index", data=test_index)
		
		new_checkpoint_dict = {}
		for i, c in enumerate(checkpoint_list):
			new_checkpoint_dict['model_link_%d' % i] = c['model_link']
		
		torch.save(new_checkpoint_dict, "../Temp/%s/Experiment_model_CV_%d" % (datetime_object, cv_count))
		checkpoint_list_all += checkpoint_list
		temp = ensemble
		try:
			temp = temp.cpu().numpy()
			test_y = test_y.cpu().numpy()
		except:
			pass
		grp.create_dataset("predicted", data=temp)
		
		grp.create_dataset("ensemble_inverse", data=scalar.inverse_transform(temp.reshape((-1, 1))).reshape((-1)))
		grp.create_dataset("test_y_inverse", data=scalar.inverse_transform(test_y.reshape((-1, 1))).reshape((-1)))
		
		cv_count += 1
	torch.save(checkpoint_list_all, "../Temp/%s/Experiment_model_list" % (datetime_object))
	f.close()
	result_file.close()
	
	
def gene_split_nn_experiments(rounds=10,n_genes=40, strict=False):
	genes_avail = np.unique(tuples)
	np.random.seed(43)
	np.random.shuffle(genes_avail)
	test_gene = genes_avail[:n_genes]
	
	# Split the train_valid / test based on genes
	if strict:
		train_valid_index, test_index = gene_split_2(tuples, test_gene)
	else:
		train_valid_index, test_index = gene_split_1(tuples, test_gene)
	
	test_y = y[test_index]
	
	

	checkpoint_list = []
	save_string_list = []
	ensemble = 0
	pool = ProcessPoolExecutor(max_workers=args.thread)
	p_list = []
	for i in range(rounds):
		print ("round %d" % i)
		index = np.random.permutation(train_valid_index)
		train_index = index[:int(0.8 * len(index))]
		valid_index = index[int(0.8 * len(index)):]
		print("train/valid/test data shape", train_index.shape, valid_index.shape, test_index.shape)
		indexs = np.array([train_index, valid_index, test_index])
		split_loc = "../Temp/%s/Experiment_%d_ind.npy" % (datetime_object, i)
		save_string = "%d" % (i)
		save_string_list.append(save_string)
		np.save(split_loc, indexs)
		p_list.append(
			pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object, save_string))
		time.sleep(60)
		

	for save_string in save_string_list:
		ckpt_list = list(torch.load("../Temp/%s/%s_model_list" % (datetime_object, save_string)))
		checkpoint_list += ckpt_list
		ensemble_part = np.load("../Temp/%s/ensemble_%s.npy" % (datetime_object, save_string))
		checkpoint_list += list(ckpt_list)
		ensemble += ensemble_part


	ensemble /= rounds
	torch.save(checkpoint_list, "../Temp/%s/Experiment_model_list" % (datetime_object))
	print("ensemble")
	pos_mask = test_y >= positive_thres
	result_file = open("../Temp/%s/result.txt" %(datetime_object), "a")
	corr1, corr2 = correlation_cuda(test_y, ensemble)
	pos_mask = test_y >= positive_thres
	corr1_pos, corr2_pos = correlation_cuda(test_y[pos_mask], ensemble[pos_mask])
	roc, aupr = roc_auc_cuda(pos_mask, ensemble, balance=True)
	print (corr1, corr2)
	print (corr1_pos, corr2_pos)
	print (roc, aupr)
	result_file.write("gene split")
	for score in [corr1, corr2, corr1_pos, corr2_pos, roc, aupr]:
		result_file.write("%f\n" %(score))
	sys.stdout.flush()
	result_file.close()
	



def whole_dataset_train(rounds=10):
	pool = ProcessPoolExecutor(max_workers=args.thread)
	p_list = []
	save_string_list = []
	for i in range(rounds):
		index = np.random.permutation( np.arange(len(tuples)))
		train_index = index[:int(0.9 * len(index))]
		valid_index = index[int(0.9 * len(index)):]
		test_index = valid_index
		indexs = np.array([train_index, valid_index, test_index])
		split_loc = "../Temp/%s/Experiment_%d_ind.npy" % (datetime_object, i)
		save_string = "%d" % (i)
		save_string_list.append(save_string)
		np.save(split_loc, indexs)
		p_list.append(
			pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object, save_string))
		time.sleep(5)
	
	pool.shutdown(wait=True)
	checkpoint_list = []
	for save_string in save_string_list:
		ckpt_list = list(torch.load("../Temp/%s/%s_model_list" % (datetime_object, save_string)))
		checkpoint_list += list(ckpt_list)
	torch.save(checkpoint_list, "../Temp/%s/Experiment_model_list" % (datetime_object))
	return


def create_hypersagnn_list(ckpt_list):
	# This is building the adjacency matrix for the GCN
	auxi_m = [np.load("../data/%s" % name) for name in
	          ['coexpression.npy', 'experimental.npy',
	           'database.npy', 'neighborhood.npy',
	           'fusion.npy', 'cooccurence.npy']]
	lambda_list = [0.1, 0.1, 1.0, 1.0, 1.0, 1.0]
	new_auxi_m = []
	for x in auxi_m:
		w = x[:, -1]
		w = MinMaxScaler(feature_range=(0, 1)).fit_transform(w.reshape((-1, 1))).reshape((len(x)))
		x[:, -1] = w
		new_auxi_m.append(x)
	auxi_m = new_auxi_m
	auxi_adj = [torch.from_numpy(build_adj_matrix(x, gene_num)).float().to(device) for x in auxi_m]
	
	hyper_sagnn_list = []
	
	for checkpoint in ckpt_list:
		graphsage_embedding, recon_nn, node_embedding, hypersagnn = get_model(gene_num, embed_dim, auxi_m, auxi_adj)
		hypersagnn.load_state_dict(checkpoint['model_link'])
		hypersagnn.node_embedding.off_hook(np.arange(1, gene_num), gene_num)
		del graphsage_embedding
		hyper_sagnn_list.append(hypersagnn)

	return hyper_sagnn_list


def ensemble_predict(hyper_sagnn_list, chunks):
	ensemble = 0
	for hypersagnn in hyper_sagnn_list:
		pred = hypersagnn.predict(chunks, verbose=True, batch_size=4800)
		ensemble += pred.detach().cpu().numpy()
	
	ensemble /= len(hyper_sagnn_list)
	pred_y = scalar.inverse_transform(ensemble.reshape((-1, 1))).reshape((-1))
	return pred_y

# all combinations of seen genes
def within_seen_genes(pos_tuple_list, hyper_sagnn_list):
	seen_genes = np.unique(pos_tuple_list.reshape((-1)))
	seen_genes = np.sort(seen_genes)
	print (len(seen_genes), seen_genes)
	
	combs = combinations(seen_genes, 3)
	total = len(seen_genes) * (len(seen_genes) - 1) * (len(seen_genes) - 2) / 6
	chunks = []
	
	finish = 0
	final_tuples = []
	final_y = []
	
	for comb in tqdm(combs, total=total):
		chunks.append(comb)
		
		if len(chunks) == 50000000:
			chunks = np.array(chunks)
			print ("finish %d of %d" %(finish,total))
			chunks = torch.from_numpy(chunks).long().to(device)
			
			pred_y = ensemble_predict(hyper_sagnn_list, chunks)
			
			selected = pred_y >= kept_thresh
			select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
			print ()
			print ("selected rate", len(select_chunks) / len(chunks))
			
			if len(select_chunks) > 0:
				final_tuples.append(select_chunks)
				final_y.append(select_pred)
				
			finish += len(chunks)
			chunks = []
		
	chunks = np.array(chunks)
	chunks = torch.from_numpy(chunks).long().to(device)
	pred_y = ensemble_predict(hyper_sagnn_list, chunks)
	
	selected = pred_y >= kept_thresh
	select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
	print(len(select_chunks))
	if len(select_chunks) > 0:
		final_tuples.append(select_chunks)
		final_y.append(select_pred)
	
	final_tuples = np.concatenate(final_tuples)
	final_y = np.concatenate(final_y)
	np.save("../Temp/%s/within_seen_tuples.npy"  % datetime_object , final_tuples)
	np.save("../Temp/%s/within_seen_y.npy"  % datetime_object, final_y)

# combinatioins of two seen genes and one unseen
def two_seen_one_unseen_genes(pos_tuple_list, hyper_sagnn_list):
	seen_genes = np.unique(pos_tuple_list.reshape((-1)))
	seen_genes = np.sort(seen_genes)
	print(len(seen_genes), seen_genes)
	seen_genes_set = set(seen_genes)
	unseen_genes = []
	for i in np.arange(1, gene_num):
		if i not in seen_genes:
			unseen_genes.append(i)
	unseen_genes = np.array(unseen_genes)
	print (len(unseen_genes), unseen_genes)
	combs = combinations(seen_genes, 2)
	total = len(seen_genes) * (len(seen_genes) - 1) / 2
	chunks = []
	
	finish = 0
	final_tuples = []
	final_y = []
	
	for comb in tqdm(combs, total=total):
		for gene in unseen_genes:
			chunks.append(list(comb) + [gene])
			
		if len(chunks) >= 50000000:
			chunks = np.array(chunks)
			print("finish %d of %d" % (finish, total))
			chunks = torch.from_numpy(chunks).long().to(device)
			
			pred_y = ensemble_predict(hyper_sagnn_list, chunks)
			
			selected = pred_y >= kept_thresh
			select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
			print()
			print("selected rate", len(select_chunks) / len(chunks))
			
			if len(select_chunks) > 0:
				final_tuples.append(select_chunks)
				final_y.append(select_pred)
			
			finish += len(chunks)
			chunks = []
	
	chunks = np.array(chunks)
	chunks = torch.from_numpy(chunks).long().to(device)
	pred_y = ensemble_predict(hyper_sagnn_list, chunks)
	
	selected = pred_y >= kept_thresh
	select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
	print(len(select_chunks))
	if len(select_chunks) > 0:
		final_tuples.append(select_chunks)
		final_y.append(select_pred)
	
	final_tuples = np.concatenate(final_tuples)
	final_y = np.concatenate(final_y)
	np.save("../Temp/%s/two_seen_one_unseen_tuples.npy" % datetime_object, final_tuples)
	np.save("../Temp/%s/two_seen_one_unseen_seen_y.npy" % datetime_object, final_y)
	
# predict on the original set
def re_evaluate(pos_tuple_list, hyper_sagnn_list):
	chunks = pos_tuple_list
	chunks = torch.from_numpy(chunks).long().to(device)
	
	pred_y = ensemble_predict(hyper_sagnn_list, chunks)
	
	selected = pred_y >= kept_thresh
	select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
	print()
	print("selected rate", len(select_chunks) / len(chunks))
	
	if len(select_chunks) > 0:
		final_tuples = select_chunks
		final_y = pred_y[selected]
	np.save("../Temp/%s/re_eval_tuples.npy"  % datetime_object, final_tuples)
	np.save("../Temp/%s/re_eval_y.npy"  % datetime_object, final_y)



args = parse_args()
# Basic parameters for training
get_free_gpu()
save_path = "../data/model_" + randomString()
embed_dim = 128
positive_thres = 0.05
kept_thresh = 0.045
positive_thres_origin = positive_thres

# Start loading data
genename2id = np.load("../data/gene2id.npy", allow_pickle=True).item()
print (np.min(list(genename2id.values())), np.max(list(genename2id.values())))
gene_num = int(np.max(list(genename2id.values())) + 1)
print ("gene_num", gene_num)

tuples = np.load("../data/tuples.npy").astype('int')
y = np.load("../data/y.npy").astype('float32')
significance = np.load("../data/sign.npy").astype('float32')

# We take the negative of y, because most of the y are negative,
# by doing so, we could directly use the prediction from regression to calculate aupr without times it with -1
y = -y
print ("num y that pases thres", np.sum(np.abs(y) > positive_thres),np.sum(y > positive_thres), np.sum(-y > positive_thres), len(y))
print ("tuples, min, max", tuples, np.min(tuples), np.max(tuples))
print ("min, max of y", np.min(y), np.max(y))
scalar = StandardScaler().fit(y.reshape((-1,1)))
y = scalar.transform(y.reshape((-1,1))).reshape((-1))
# Remember to transform the positive threshold too
positive_thres = float(scalar.transform(np.array([positive_thres]).reshape((-1, 1)))[0])


main_task_loss_func = [log_cosh]

reconstruct_loss_func = sparse_mse

datetime_object = args.identifier
if datetime_object is None:
	datetime_object = str(datetime.datetime.now())
	datetime_object = "_".join(datetime_object.split(":")[:-1])
	datetime_object = datetime_object.replace(" ", "_")
	datetime_object += args.extra
	

if os.path.exists("../Temp/%s" % datetime_object):
	shutil.rmtree("../Temp/%s" % datetime_object)
os.mkdir("../Temp/%s" % datetime_object)

if args.mode == 'eval':
	if args.split == 0:
		random_cv_nn_experiments(5, 10)
	elif args.split == 1:
		gene_split_nn_experiments(10, 40, False)
	elif args.split == 2:
		gene_split_nn_experiments(10, 400, True)
elif args.mode == 'train':
	whole_dataset_train(10)
elif args.mode == 'predict':
	ckpt_list = list(torch.load("../Temp/%s/Experiment_model_list" % (datetime_object), map_location='cpu'))
	hyper_sagnn_list = create_hypersagnn_list(ckpt_list)
	if args.predict == 0:
		re_evaluate(tuples, hyper_sagnn_list)
	elif args.predict == 0:
		within_seen_genes(tuples, hyper_sagnn_list)
	elif args.predict == 0:
		two_seen_one_unseen_genes(tuples, hyper_sagnn_list)
else:
	print ("Unknown mode")




