import torch
import os
import pandas as pd

# os.environ["device"] = 'gpu' # default value (this will be overridden)
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="DANGO Main Program")
	parser.add_argument('-t', '--thread', type=int, default=8)
	parser.add_argument('-m', '--mode', type=str, default='train')
	parser.add_argument('-i', '--identifier', type=str, default=None)
	parser.add_argument('-f', '--modelfolder', type=str, default=None, help="add folder of pretrained model to be used")
	parser.add_argument('-s', '--split', type=int, default=0)
	parser.add_argument('-p', '--predict', type=int, default=0)
	parser.add_argument('-d', '--device', type=int, default='0', help="-1 if no gpu")
	parser.add_argument('--gp', action="store_true", help="add gaussian process")
	parser.add_argument('-w', '--withprotein', type=str,default=None, help="supply plath to protein embeddings")
	return parser.parse_args()

args = parse_args()
os.environ["device"] = 'cpu' if args.device == -1 else str(args.device)
device = os.environ["device"]

from model import get_model
from train import *
import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split

import h5py
from GPR import *

import subprocess
from itertools import combinations
import random
import shutil
# This function is splitting data to make all test data nodes are unseen in training




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


def id2baselinevecs(vecs, ids, type="cat"):
    vecs_selected = vecs[ids - 1]  # shape: [B, K, D]

    if type == "avg":
        return vecs_selected.mean(axis=1)  # [B, D]
    elif type == "cat":
        return vecs_selected.reshape(vecs_selected.shape[0], -1)  # [B, K*D]
    else:
        raise ValueError(f"Unknown type: {type}. Use 'avg' or 'cat'")



def baseline(train_data, train_y, test_data, test_y, identifier="", random_shuffle=False, agg="cat"):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    result_file = open("../Temp/%s/result.txt" % (datetime_object), "w")
    def log(s):
        print(s)
        print(s, file=result_file)

    log(f"start baseline with aggregation type {agg}")
    baseline_vec = np.load("../data/embeddings.npy")
    if random_shuffle:
        np.random.shuffle(baseline_vec)

    baseline_feature_train = id2baselinevecs(baseline_vec, train_data, type=agg)
    baseline_feature_test = id2baselinevecs(baseline_vec, test_data, type=agg)

    # ---------- XGBoost ----------
    xgb = XGBRegressor(n_jobs=cpu_num, n_estimators=500, max_depth=10).fit(baseline_feature_train, train_y)
    y_pred_xgb = xgb.predict(baseline_feature_test)

    corr_all_xgb = correlation_cuda(test_y, y_pred_xgb)
    pos_mask = test_y >= positive_thres
    corr_pos_xgb = correlation_cuda(test_y[pos_mask], y_pred_xgb[pos_mask])
    auc_xgb = roc_auc_cuda(pos_mask, y_pred_xgb, balance=True)

    log("xgboost baseline")
    log(f"corr (all): {corr_all_xgb}")
    log(f"corr (pos): {corr_pos_xgb}")
    log(f"roc_auc: {auc_xgb}")

    # ---------- Random Forest ----------
    rf = RandomForestRegressor(n_jobs=cpu_num, n_estimators=500, max_depth=10).fit(baseline_feature_train, train_y)
    y_pred_rf = rf.predict(baseline_feature_test)

    corr_all_rf = correlation_cuda(test_y, y_pred_rf)
    corr_pos_rf = correlation_cuda(test_y[pos_mask], y_pred_rf[pos_mask])
    auc_rf = roc_auc_cuda(pos_mask, y_pred_rf, balance=True)

    log(f"rf baseline {identifier}:")
    log(f"corr (all): {corr_all_rf}")
    log(f"corr (pos): {corr_pos_rf}")
    log(f"roc_auc: {auc_rf}")

    result_file.close()

    return {
        "xgb_pearson_all": corr_all_xgb[0],
        "xgb_spearman_all": corr_all_xgb[1],
        "xgb_pearson_pos": corr_pos_xgb[0],
        "xgb_spearman_pos": corr_pos_xgb[1],
        "xgb_auroc": auc_xgb[0],
        "xgb_auprc": auc_xgb[1],
        "rf_pearson_all": corr_all_rf[0],
        "rf_spearman_all": corr_all_rf[1],
        "rf_pearson_pos": corr_pos_rf[0],
        "rf_spearman_pos": corr_pos_rf[1],
        "rf_auroc": auc_rf[0],
        "rf_auprc": auc_rf[1],
    }


# call subprocess for training a Dango instance (used in multiprocessing)
def mp_train(positive_thres, gene_num, split_loc, save_dir, save_string, withPPI):
	cmd = ["python", "train.py", str(positive_thres), str(gene_num), str(embed_dim), split_loc, save_dir, save_string, str(withPPI)]
	subprocess.call(cmd)
	print("start reading")


# def fit_gaussian_process(triplet_embeddings, residuals):
# 	print("Fitting Gaussian Process...")
# 	kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# 	GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, copy_X_train=False)
# 	GP.fit(triplet_embeddings, residuals)  # fit on the residuals of DANGO
# 	optimized_kernel = GP.kernel_
# 	print(f'GP Optimized Kernel: %s' % str(optimized_kernel))
#
# 	return GP, optimized_kernel

def random_cv_nn_experiments(n_fold=5, rounds=10,debug=False, withPPI=False, run_baseline=False):
	print("random_cv_params", n_fold, rounds)
	kf = KFold(n_splits=n_fold, shuffle=True)
	f = h5py.File(os.path.join("../Temp/%s/Experiment.hdf5" % datetime_object), "w")
	result_file = open("../Temp/%s/result.txt" % (datetime_object), "a")
	cv_count = 0
 
	if run_baseline:
		results_avg = []
		results_cat = []

		for train_index, test_index in kf.split(tuples):
			train_data = tuples[train_index]
			train_y = y[train_index]
			test_data = tuples[test_index]
			test_y = y[test_index]

			res_avg = baseline(train_data=train_data, train_y=train_y, test_data=test_data, test_y=test_y, agg="avg")
			res_cat = baseline(train_data=train_data, train_y=train_y, test_data=test_data, test_y=test_y, agg="cat")

			results_avg.append(res_avg)
			results_cat.append(res_cat)

		df_avg = pd.DataFrame(results_avg)
		df_cat = pd.DataFrame(results_cat)
		mean_avg = df_avg.mean()
		mean_cat = df_cat.mean()

		print("\nAverage Metrics (agg='avg'):", file=result_file)
		print(mean_avg, file=result_file)

		print("\nAverage Metrics (agg='cat'):", file=result_file)
		print(mean_cat, file=result_file)

	else:
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
				
				indexs = np.array([train_index, valid_index, test_index], dtype=object)
				split_loc = "../Temp/%s/Experiment_%d_%d_ind.npy" % (datetime_object, cv_count, i)
				save_string = "%d_%d" % (cv_count, i)
				save_string_list.append(save_string)
				np.save(split_loc, indexs)
				if debug:
					mp_train(positive_thres_origin, gene_num, split_loc, datetime_object,save_string)
				else:
					p_list.append(pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object,
											save_string, withPPI))
					time.sleep(60)
			pool.shutdown(wait=True)
			finish_count = 0
			print(save_string_list)
			
			for save_string in save_string_list:
				print("Now at Fold %d iteration %d" % (cv_count, finish_count))
				# Some child process might get killed due to memory/GPU memory usage
				# Or someone you are sharing the machine with decide to run a gigantic process in the middle
				# The try / exception here would skip those killed child process
				# so that you won't get frustrated about running the 5-fold CV from scratch
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

			if finish_count == 0:
				finish_count = 1
			ensemble /= finish_count

			print("ensemble at Fold %d" % cv_count)
			corr1, corr2 = correlation_cuda(test_y, ensemble)
			pos_mask = test_y >= positive_thres
			corr1_pos, corr2_pos = correlation_cuda(test_y[pos_mask], ensemble[pos_mask])
			roc, aupr = roc_auc_cuda(pos_mask, ensemble, balance=True)
			print("Pearson, Spearman")
			print(corr1, corr2)
			print("Pearson_strong, Spearman_strong")
			print(corr1_pos, corr2_pos)
			print("AUC, AUPR")
			print(roc, aupr)
			sys.stdout.flush()

			corr1, corr2 = correlation_cuda(test_y, ensemble)
			pos_mask = test_y >= positive_thres
			corr1_pos, corr2_pos = correlation_cuda(test_y[pos_mask], ensemble[pos_mask])
			roc, aupr = roc_auc_cuda(pos_mask, ensemble, balance=True)
			result_file.write("Fold %d\n" % (cv_count))
			metrics = [corr1, corr2, corr1_pos, corr2_pos, roc, aupr]
			for score in metrics:
				print("writing score:" + str(score))
				result_file.write("%f\n" % score)
				result_file.flush()
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
		result_file.close()
		f.close()
		# return metrics

	
def gene_split_nn_experiments(rounds=10, n_genes=40, strict=False, withPPI=False, run_baseline=False):
	genes_avail = np.unique(tuples)
	np.random.seed(43)
	np.random.shuffle(genes_avail)
	test_gene = genes_avail[:n_genes]
	
	result_file = open("../Temp/%s/result.txt" % (datetime_object), "a")
 
	# Split the train_valid / test based on genes
	if strict:
		train_valid_index, test_index = gene_split_2(tuples, test_gene)
	else:
		train_valid_index, test_index = gene_split_1(tuples, test_gene)
	
	test_y = y[test_index]
	
	
	if run_baseline:

		train_data = tuples[train_valid_index]
		train_y = y[train_valid_index]
		test_data = tuples[test_index]
		test_y = y[test_index]

		res_avg = baseline(train_data=train_data, train_y=train_y, test_data=test_data, test_y=test_y, agg="avg")
		res_cat = baseline(train_data=train_data, train_y=train_y, test_data=test_data, test_y=test_y, agg="cat")

		print("\nAverage Metrics (agg='avg'):", file=result_file)
		print(res_avg, file=result_file)

		print("\nAverage Metrics (agg='cat'):", file=result_file)
		print(res_cat, file=result_file)
  
	else:
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
			indexs = np.array([train_index, valid_index, test_index], dtype=object)
			split_loc = "../Temp/%s/Experiment_%d_ind.npy" % (datetime_object, i)
			save_string = "%d" % (i)
			save_string_list.append(save_string)
			np.save(split_loc, indexs)
			p_list.append(
				pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object, save_string, withPPI))
			time.sleep(60)
		pool.shutdown(wait=True)

		for save_string in save_string_list:
			try:
				ckpt_list = list(torch.load("../Temp/%s/%s_model_list" % (datetime_object, save_string)))
				checkpoint_list += ckpt_list
				ensemble_part = np.load("../Temp/%s/ensemble_%s.npy" % (datetime_object, save_string))
				checkpoint_list += list(ckpt_list)
				ensemble += ensemble_part
			except:
				pass


	ensemble /= rounds
	torch.save(checkpoint_list, "../Temp/%s/Experiment_model_list" % (datetime_object))
	print("ensemble")
	pos_mask = test_y >= positive_thres
	corr1, corr2 = correlation_cuda(test_y, ensemble)
	pos_mask = test_y >= positive_thres
	corr1_pos, corr2_pos = correlation_cuda(test_y[pos_mask], ensemble[pos_mask])
	roc, aupr = roc_auc_cuda(pos_mask, ensemble, balance=True)
	print ("Pearson, Spearman")
	print (corr1, corr2)
	print("Pearson_strong, Spearman_strong")
	print (corr1_pos, corr2_pos)
	print ("AUC, AUPR")
	print (roc, aupr)
	sys.stdout.flush()
	metrics = [corr1, corr2, corr1_pos, corr2_pos, roc, aupr]
	print(metrics)
	split = 1 if not strict else 2
	with open(f"../Temp/{datetime_object}/result_split{split}.txt", "a") as result_file:
		result_file.write("gene split\n")
		for score in metrics:
			result_file.write("%f\n" %(score))
			result_file.flush()
	# result_file.close()
	sys.stdout.flush()
	return metrics


def whole_dataset_train(rounds=10, withPPI=False):
	pool = ProcessPoolExecutor(max_workers=args.thread)
	p_list = []
	save_string_list = []
	for i in range(rounds):
		index = np.random.permutation( np.arange(len(tuples)))
		train_index = index[:int(0.9 * len(index))]
		valid_index = index[int(0.9 * len(index)):]
		test_index = valid_index
		indexs = np.array([train_index, valid_index, test_index], dtype=object)
		print(indexs)
		split_loc = "../Temp/%s/Experiment_%d_ind.npy" % (datetime_object, i)
		save_string = "%d" % (i)
		save_string_list.append(save_string)
		np.save(split_loc, indexs)
		p_list.append(
			pool.submit(mp_train, positive_thres_origin, gene_num, split_loc, datetime_object, save_string, withPPI))
		time.sleep(5)
	
	pool.shutdown(wait=True)
	checkpoint_list = []
	for save_string in save_string_list:
		ckpt_list = list(torch.load("../Temp/%s/%s_model_list" % (datetime_object, save_string)))
		checkpoint_list += list(ckpt_list)
	torch.save(checkpoint_list, "../Temp/%s/Experiment_model_list" % (datetime_object))
	return


def create_dango_list(ckpt_list):
	# This is building the adjacency matrix for the GCN
	auxi_m = [np.load("../data/%s" % name) for name in
			  ['coexpression.npy', 'experimental.npy',
			   'database.npy', 'neighborhood.npy',
			   'fusion.npy', 'cooccurence.npy']]
	# lambda_list = [0.1, 0.1, 1.0, 1.0, 1.0, 1.0]
	lambda_list = [0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
	new_auxi_m = []
	for x in auxi_m:
		w = x[:, -1]
		w = MinMaxScaler(feature_range=(0, 1)).fit_transform(w.reshape((-1, 1))).reshape((len(x)))
		x[:, -1] = w
		new_auxi_m.append(x)
	auxi_m = new_auxi_m
	auxi_adj = [torch.from_numpy(build_adj_matrix(x, gene_num)).float().to(device) for x in auxi_m]
	
	dango_list = []
	
	for checkpoint in ckpt_list:
		graphsage_embedding, recon_nn, node_embedding, hypersagnn, ppi_nn = get_model(gene_num, embed_dim, auxi_m, auxi_adj)
		hypersagnn.load_state_dict(checkpoint['model_link'])
		hypersagnn.node_embedding.off_hook(np.arange(1, gene_num), gene_num)
		del graphsage_embedding
		dango_list.append(hypersagnn)

	return dango_list


def ensemble_predict(dango_list, chunks):
	ensemble = 0
	for hypersagnn in dango_list:
		pred = hypersagnn.predict(chunks, verbose=True, batch_size=4800)
		ensemble += pred.detach().cpu().numpy()
	
	ensemble /= len(dango_list)
	pred_y = scalar.inverse_transform(ensemble.reshape((-1, 1))).reshape((-1))
	return pred_y


# get original node embeddings that are inputs to the model
def get_embeddings(dango_list, chunks, batch_size=4800):# add permutation of embeddings
	ensemble_embeddings = 0
	for hypersagnn in dango_list:
		batch_outputs = []
		hypersagnn.eval()
		with torch.no_grad():
			for j in range(math.ceil(len(chunks) / batch_size)):
				x = chunks[j * batch_size:min((j + 1) * batch_size, len(chunks))]
				dynamic, static, _ = hypersagnn.get_embedding(x)
				batch_embedding = (dynamic - static) ** 2
				batch_outputs.append(dynamic)
			embedding = torch.cat(batch_outputs, dim=0)
			ensemble_embeddings += torch.mean(embedding, dim=1)
	ensemble_embeddings /= len(dango_list)
	return ensemble_embeddings


def ensemble_predict_gp(dango_list, chunks, GP, embeddings):
	pred_y = ensemble_predict(dango_list, chunks)
	residual, sigma = GP.predict(embeddings, return_std=True)

	return pred_y + residual, sigma


# all combinations of seen genes
def within_seen_genes(pos_tuple_list, dango_list):
	seen_genes = np.unique(pos_tuple_list.reshape((-1)))
	seen_genes = np.sort(seen_genes)
	print(len(seen_genes), seen_genes)
	
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
			print("finish %d of %d" % (finish, total))
			chunks = torch.from_numpy(chunks).long().to(device)
			
			pred_y = ensemble_predict(dango_list, chunks)
			
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

	# means = None
	# variances = None
	# if GP is not None:
	# 	gp_pred_x = get_embeddings(dango_list, chunks).to(device)
	# 	predict_dataset = TensorDataset(gp_pred_x, torch.zeros(gp_pred_x.size(dim=0),1))
	# 	predict_loader = DataLoader(predict_dataset, batch_size=1024, shuffle=False)
	# 	means, variances = predict_variational_gp(predict_loader, GP, likelihood)

	pred_y = ensemble_predict(dango_list, chunks)


	
	selected = pred_y >= kept_thresh
	select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
	# select_means, select_variances = means[selected], variances[selected]
	print(len(select_chunks))
	if len(select_chunks) > 0:
		final_tuples.append(select_chunks)
		final_y.append(select_pred)
	
	final_tuples = np.concatenate(final_tuples)
	final_y = np.concatenate(final_y)
	np.save("../Temp/%s/within_seen_tuples.npy" % datetime_object, final_tuples)
	np.save("../Temp/%s/within_seen_y.npy" % datetime_object, final_y)
	# if means is not None and variances is not None:
	# 	np.save("../Temp/%s/within_seen_gp_means.npy" % datetime_object, select_means)
	# 	np.save("../Temp/%s/within_seen_gp_vars.npy" % datetime_object, select_variances)


# combinations of two seen genes and one unseen
def two_seen_one_unseen_genes(pos_tuple_list, dango_list):
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
			
			pred_y = ensemble_predict(dango_list, chunks)
			
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
	pred_y = ensemble_predict(dango_list, chunks)
	
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
def re_evaluate(pos_tuple_list, dango_list):
	chunks = pos_tuple_list
	chunks = torch.from_numpy(chunks).long().to(device)
	pred_y = ensemble_predict(dango_list, chunks)

	kept_thresh = 0.05
	selected = pred_y >= kept_thresh
	select_chunks, select_pred = chunks[selected].detach().cpu().numpy(), pred_y[selected]
	print()
	print("selected rate", len(select_chunks) / len(chunks))

	# if len(select_chunks) > 0:
	final_tuples = select_chunks
	final_y = pred_y[selected]
	np.save("../Temp/%s/re_eval_tuples.npy" % datetime_object, final_tuples)
	np.save("../Temp/%s/re_eval_y.npy" % datetime_object, final_y)


if __name__ == '__main__':
	# Basic parameters for training
	# devices = get_free_gpus(2)
	# devices = [7]
	devices = [int(os.environ["device"])]
	os.environ["protein_embedding_path"] = args.withprotein if args.withprotein else ""
	withprotein = True if args.withprotein else False
 
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

	if not os.path.exists("../Temp/%s" % datetime_object):
		os.mkdir("../Temp/%s" % datetime_object)
      
	args.mode = args.mode.split(";")
	if 'eval-debug' in args.mode:
		if args.split == 0:
			print("in eval-debug")
			random_cv_nn_experiments(2, 1, debug=True, withPPI=withprotein)
	elif 'eval' in args.mode:
		metrics = None
		if args.split == 0:
			print("Evaluating on split 0")
			random_cv_nn_experiments(5, 1,withPPI=withprotein)
		elif args.split == 1:
			print("Evaluating on split 1")
			gene_split_nn_experiments(10, 40, False, withPPI=withprotein)
		elif args.split == 2:
			print("Evaluating on split 2")
			gene_split_nn_experiments(10, 400, True, withPPI=withprotein)
		else:
			pass
	elif 'train' in args.mode:
		whole_dataset_train(10)
	elif "baseline" in args.mode:
		if args.split == 0:
			print("Evaluating on split 0")
			random_cv_nn_experiments(2, 1,withPPI=withprotein, run_baseline=True)
		elif args.split == 1:
			print("Evaluating on split 1")
			gene_split_nn_experiments(10, 40, False, withPPI=withprotein,  run_baseline=True)
		elif args.split == 2:
			print("Evaluating on split 2")
			gene_split_nn_experiments(10, 400, True, withPPI=withprotein,  run_baseline=True)
   
	elif 'predict' in args.mode:
		if args.modelfolder is not None and os.path.exists("../Temp/%s" % args.modelfolder):
			ckpt_list = list(torch.load("../Temp/%s/Experiment_model_list" % args.modelfolder, map_location='cpu'))
		else:
			ckpt_list = list(torch.load("../Temp/%s/Experiment_model_list" % datetime_object, map_location='cpu'))
		dango_list = create_dango_list(ckpt_list)

		if args.predict == 0:
			print("Predicting on split 0")
			re_evaluate(tuples, dango_list)
		elif args.predict == 1:
			print("Predicting on split 1")
			within_seen_genes(tuples, dango_list)
		elif args.predict == 2:
			print("Predicting on split 2")
			two_seen_one_unseen_genes(tuples, dango_list)
		else:
			pass

		if args.gp and args.predict != 0:

			# --------------------------------------- Training Variational GP --------------------------------- #

			tuples_name = "within_seen_tuples.npy" if args.predict == 1 else "two_seen_one_unseen_tuples.npy"

			from torch.utils.data import TensorDataset, DataLoader
			chunks = torch.from_numpy(tuples).long().to(devices[0])
			x = get_embeddings(dango_list, chunks).contiguous().to(devices[0])
			y = torch.from_numpy(y).contiguous().to(devices[0])
			pred_y = torch.from_numpy(ensemble_predict(dango_list, chunks)).to(devices[0])

			train_x = x
			train_y = y - pred_y
			np.save("../Temp/%s/residuals.npy" % datetime_object, train_y.cpu().numpy())

			train_dataset = TensorDataset(train_x, train_y)
			train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

			GP, likelihood = train_variational_gp(train_loader, train_x, train_y, devices[0])
			torch.save(GP.state_dict(), "../Temp/%s/Trained_GaussianProcess" % datetime_object)

			torch.cuda.empty_cache()
			del chunks, x, y, train_x, train_y, train_loader, pred_y

			# ------------------------------------ Predicting with Variational GP ----------------------------- #
			tuples = torch.from_numpy(np.load("../Temp/%s/%s" % (datetime_object, tuples_name))).to(devices[0])
			predict_dataset = TensorDataset(tuples, torch.zeros(tuples.size(dim=0), 1))		# add useless target value
			predict_loader = DataLoader(predict_dataset, batch_size=10000, shuffle=False)
			means, variances = predict_variational_gp(predict_loader, GP.to(devices[0]), likelihood.to(devices[0]), dango_list, get_embeddings, devices[0])

			print(len(tuples), len(means), len(variances))
			np.save("../Temp/%s/gp_means.npy" % datetime_object, means)
			np.save("../Temp/%s/gp_variances.npy" % datetime_object, variances)

	else:
		print("Unknown mode")


# predict = False
	# if predict:
	# 	chunks = torch.from_numpy(tuples).long().to(devices[0])
	# 	embeddings = get_embeddings(dango_list, chunks).contiguous().to(devices[0])
	# 	chunks = chunks.cpu()
	# 	train_y = torch.from_numpy(y).contiguous().to(devices[0])
	# 	state_dict = torch.load('../Temp/2022-07-05_09_50/GaussianProcess')
	# 	gp = ExactGPModel(embeddings, train_y, device_ids=devices, output_device=devices[0])  # Create a new GP model
	# 	gp.load_state_dict(state_dict)
	# 	torch.cuda.empty_cache()
	# 	predictions = predict_gp(gp.to(devices[0]), embeddings.to('cpu'), devices=devices)

	# ----------------------------------------------CPU GP------------------------------------------- #

	# chunks = torch.from_numpy(tuples).long().to(devices[0])
	# x = get_embeddings(dango_list, chunks).to('cpu')
	# print(type(x))
	# mask = np.random.rand(len(x)) < 0.8
	# test_x = x[~mask]
	# pred_y = torch.from_numpy(ensemble_predict(dango_list, chunks)).to(devices[0])
	# train_y = torch.from_numpy(y).to(devices[0])
	# y = (train_y-pred_y).to('cpu')
	# likelihood = gpytorch.likelihoods.GaussianLikelihood()
	# model = CpuGP(x, y, likelihood)
	# model.train()
	# likelihood.train()
	# print(x)
	# train_gp_cpu(model, likelihood, x, y, training_iter=20)
	#
	# model.eval()
	# likelihood.eval()
	# # Make predictions by feeding model through likelihood
	# with torch.no_grad(), gpytorch.settings.fast_pred_var():
	# 	# Test points are regularly spaced along [0,1]
	# 	predictions = likelihood(model(test_x))
	# print(predictions)
	# ------------------------------------------------------------------------------------------------- #
# means, variances = predict_variational_gp(test_loader, GP, likelihood)

# selected = pred_y.cpu().numpy() >= kept_thresh
# select_chunks, select_pred = train_x[selected].detach().cpu().numpy(), pred_y[selected].cpu().numpy()
# selected_means, selected_vars = means[selected], variances[selected]
# print()
# print("selected rate", len(select_chunks) / len(chunks))
#
# # if len(select_chunks) > 0:
# # 	final_tuples = select_chunks
# # 	final_y = pred_y[selected]
# # np.save("../Temp/%s/gp_tuples.npy" % datetime_object, select_chunks)
# np.save("../Temp/%s/dango_preds_y.npy" % datetime_object, select_pred)
# np.save("../Temp/%s/gp_means.npy" % datetime_object, selected_means)
# np.save("../Temp/%s/gp_variances.npy" % datetime_object, selected_vars)

# if args.gaussian:
# 	print("Preprocessing GP training data...")
# 	chunks = torch.from_numpy(tuples).long().to(devices[0])
# 	x = get_embeddings(dango_list, chunks).contiguous().to(devices[0])
# 	mask = np.random.rand(len(x)) < 0.8
# 	# train_x, test_x = x[mask], x[~mask]
# 	pred_y = torch.from_numpy(ensemble_predict(dango_list, chunks)).to(devices[0])
# 	print(pred_y)
# 	train_x = x
# 	train_y = torch.from_numpy(y).contiguous().to(devices[0])
# 	print("Fitting Gaussian Process...")
# 	GP, _ = fit_gaussian_process(train_x, train_y-pred_y)
#
# 	train_y = train_y.detach().cpu().numpy()
# 	pred_y = pred_y.detach().cpu().numpy()
# 	del pred_y
# 	del train_y
# 	print("Evaluating Gaussian Process...")
# 	GP.eval()
# 	torch.save(GP.state_dict(), "../Temp/%s/GaussianProcess" % (datetime_object))
# 	# likelihood.eval()
# 	train_x = train_x.detach().cpu()
# 	# torch.cuda.list_gpu_processes("cuda:%d"%device)
# 	# torch.cuda.mem_get_info("cuda:%d"%device)
# 	# torch.cuda.memory_summary("cuda:%d"%device)
# 	batch_size = 100
# 	pred_means = []
# 	pred_vars = []
# 	with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(35):
# 		for j in range(math.ceil(len(train_x) / batch_size)):
# 			x = train_x[j * batch_size:min((j + 1) * batch_size, len(train_x))].to(device)
# 			print(x.shape)
# 			prediction = GP(x)
# 			mean = prediction.mean.detach().cpu().numpy()
# 			var = prediction.variance.detach().cpu().numpy()
# 			pred_means.append(mean)
# 			pred_vars.append(var)
# 		pred_means = torch.cat(pred_means, dim=0)
# 		pred_vars = torch.cat(pred_vars, dim=0)
# 	pred_vars = pred_vars.detach().cpu().numpy()
# 	pred_means = pred_means.detach().cpu().numpy()
# 	print(pred_means)
# 	print(pred_means.flatten())
# 	print(pred_vars.flatten())
# 	print(pred_means.flatten() + pred_y.detach().cpu().numpy())
# 	selected = pred_y >= kept_thresh
# 	select_chunks, select_pred = train_x[selected].detach().cpu().numpy(), pred_y[selected]
# 	selected_means, selected_vars = pred_means[selected], pred_vars[selected]
# 	print()
# 	print("selected rate", len(select_chunks) / len(chunks))
#
# 	# if len(select_chunks) > 0:
# 	# 	final_tuples = select_chunks
# 	# 	final_y = pred_y[selected]
# 	np.save("../Temp/%s/re_eval_tuples.npy" % datetime_object, select_chunks)
# 	np.save("../Temp/%s/re_eval_y.npy" % datetime_object, select_pred)
# 	np.save("../Temp/%s/re_eval_selected_means.npy" % datetime_object, selected_means)
# 	np.save("../Temp/%s/re_eval_selected_vars.npy" % datetime_object, selected_vars)
	
	
