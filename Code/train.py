import torch
import os
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Modules import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import *
import sys
from train import *
import time


def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		ids = np.where(memory_available == max_mem)[0]
		chosen_id = int(np.random.choice(ids, 1)[0])
		print("setting to gpu:%d" % chosen_id)
		torch.cuda.set_device(chosen_id)
	else:
		return

# Forward process of the hyper-sagnn
def forward_batch_hyper(model, batch_data, y, sample_weight=None):
	x = batch_data
	pred = model(x).view(-1)
	loss = 0.0
	for loss_ in main_task_loss_func:
		loss += loss_(pred, y, sample_weight)
	return pred, loss

# forward process of reconstruction
def forward_batch_recon(embed_nn, recon_nn, batch_data, adj, lambda_=1.0):
	x = batch_data.view(-1)
	pred = recon_nn(embed_nn(x))
	
	loss = reconstruct_loss_func(pred, adj[x - 1], lambda_=lambda_)
	return loss

# Training process for one epoch
def train_epoch(model, embed_nn_list, recon_nn_list, data, recon_adj_list, optimizer, batch_size, device):
	if len(data) == 2:
		tuples, y = data
		sample_weight = torch.ones_like(y)
	else:
		tuples, y, sample_weight = data
		
	# Permutate all the data
	index = torch.randperm(len(tuples))[:10000]
	tuples, y, sample_weight = tuples[index], y[index], sample_weight[index]
	
	model.train()
	
	total_loss_main = 0
	total_loss_recon = 0
	batch_num = int(math.floor(len(tuples) / batch_size))
	bar = trange(batch_num, desc=' - (Train) ', leave=False, )
	
	y_list, pred_list = [], []
	for i in bar:
		batch_data = tuples[i * batch_size:(i + 1) * batch_size]
		batch_y = y[i * batch_size:(i + 1) * batch_size]
		batch_sw = sample_weight[i * batch_size:(i + 1) * batch_size]
		batch_recon_index = torch.randint(low=1, high=gene_num, size=[batch_size * 10, ], device=device)
		pred, loss_regress = forward_batch_hyper(model, batch_data, y=batch_y, sample_weight=batch_sw)
		y_list.append(batch_y)
		pred_list.append(pred)
		loss_recon = 0
		
		for j, embed_nn, recon_nn, recon_adj in enumerate(zip(embed_nn_list, recon_nn_list, recon_adj_list)):
			loss_recon += forward_batch_recon(embed_nn, recon_nn, batch_recon_index, recon_adj, lambda_list[j])
		
		beta = 0.001 if isinstance(model, Hyper_SAGNN) else 0
		loss = loss_regress + loss_recon * beta
		
		optimizer.zero_grad()
		
		# backward
		loss.backward()
		
		# update parameters
		optimizer.step()
		
		bar.set_description(" - (Train):  %.4f Recon: %.4f " %
		                    (total_loss_main / (i + 1), total_loss_recon / (i + 1)), refresh=False)
		total_loss_main += loss_regress.item()
		total_loss_recon += loss_recon.item()
	
	y = torch.cat(y_list)
	pred = torch.cat(pred_list)
	
	# Transform the predicted value and ground truth to the original scale before calculating the metrics
	# Because the threshold to define positive is still on the orignal scale
	temp_y = torch.from_numpy(scalar.inverse_transform(y.detach().cpu().numpy().reshape((-1, 1)))).view(-1).to(device)
	temp_pred = torch.from_numpy(scalar.inverse_transform(pred.detach().cpu().numpy().reshape((-1, 1)))).view(-1).to(
		device)
	corr1, corr2 = correlation_cuda(temp_y, temp_pred)
	pos_mask = (temp_y >= positive_thres)
	corr1_pos, corr2_pos = correlation_cuda(temp_y[pos_mask],
	                                        temp_pred[pos_mask])
	roc, aupr = roc_auc_cuda(pos_mask.float(),
	                         temp_pred, balance=True)
	
	return total_loss_main / (i + 1), total_loss_recon / (i + 1), corr1, corr2, corr1_pos, corr2_pos, roc, aupr

# Evaluation process for one epoch
def eval_epoch(model, embed_nn_list, recon_nn_list, data, recon_adj_list, optimizer, batch_size, device):
	if len(data) == 2:
		tuples, y = data
		sample_weight = torch.ones_like(y)
	else:
		tuples, y, sample_weight = data
	
	# Permutate all the data
	index = torch.randperm(len(tuples))
	tuples, y, sample_weight = tuples[index], y[index], sample_weight[index]
	
	model.eval()
	with torch.no_grad():
		total_loss_main = 0
		total_loss_recon = 0
		batch_num = int(math.ceil(len(tuples) / batch_size))
		bar = trange(batch_num, desc=' - (Eval) ', leave=False, )
		
		y_list, pred_list = [], []
		for i in bar:
			batch_data = tuples[i * batch_size:(i + 1) * batch_size]
			batch_y = y[i * batch_size:(i + 1) * batch_size]
			
			pred, loss_regress = forward_batch_hyper(model, batch_data, y=batch_y, sample_weight=sample_weight[i * batch_size:(i + 1) * batch_size])
			
			y_list.append(batch_y)
			pred_list.append(pred)
			loss_recon = 0
			
			for embed_nn, recon_nn, recon_adj in zip(embed_nn_list, recon_nn_list, recon_adj_list):
				loss_recon += forward_batch_recon(embed_nn, recon_nn, batch_data, recon_adj)
			
			bar.set_description(" - (Eval):  %.4f Recon: %.4f " %
			                    (total_loss_main / (i + 1), total_loss_recon / (i + 1)), refresh=False)
			total_loss_main += loss_regress.item()
			total_loss_recon += loss_recon.item()
		
		y = torch.cat(y_list)
		pred = torch.cat(pred_list)
		
		if scalar is not None:
			temp_y = torch.from_numpy(scalar.inverse_transform(y.detach().cpu().numpy().reshape((-1, 1)))).view(-1).to(
				device)
			temp_pred = torch.from_numpy(scalar.inverse_transform(pred.detach().cpu().numpy().reshape((-1, 1)))).view(
				-1).to(
				device)
		else:
			temp_y = y
			temp_pred = pred
		
		corr1, corr2 = correlation_cuda(temp_y, temp_pred)
		pos_mask = temp_y >= positive_thres
		corr1_pos, corr2_pos = correlation_cuda(temp_y[pos_mask],
		                                        temp_pred[pos_mask])
		roc, aupr = roc_auc_cuda(pos_mask.float(),
		                         temp_pred, balance=True)
	return total_loss_main / (i + 1), total_loss_recon / (i + 1), corr1, corr2, corr1_pos, corr2_pos, roc, aupr


def train(model, embed_nn_list, recon_nn_list, training_data, validation_data, test_data, recon_adj_list, optimizer,
          epochs, batch_size, device):
	checkpoint_list = []
	checkpoint_score = []
	ensemble_num = 5
	
	valid_metric = [0.0]
	no_change = 0
	for epoch_i in range(epochs):
		
		print('[ Epoch', epoch_i, 'of', epochs, ']')
		
		start = time.time()
		
		loss_regress, loss_recon, corr1, corr2, corr1_pos, corr2_pos, roc, aupr = train_epoch(
			model, embed_nn_list, recon_nn_list, training_data, recon_adj_list, optimizer, batch_size, device)
		print('  - (Train) : {loss_regress: 7.4f}, '
		      ' recon: {loss_recon:3.2f}, pearson: {corr1:3.2f}, spearman: {corr2:3.2f}, '
		      ' pearson_pos: {corr1_pos:3.2f}, spearman_pos: {corr2_pos:3.2f}, '
		      ' roc: {roc:3.2f}, aupr: {aupr:3.2f}, '
		      'elapse: {elapse:3.2f} s'.format(
			loss_regress=loss_regress,
			loss_recon=loss_recon,
			corr1=corr1,
			corr2=corr2,
			corr1_pos=corr1_pos,
			corr2_pos=corr2_pos,
			roc=roc,
			aupr=aupr,
			elapse=(time.time() - start)))
		
		start = time.time()
		valid_loss_regress, valid_loss_recon, valid_corr1, valid_corr2, valid_corr1_pos, valid_corr2_pos, valid_roc, valid_aupr = eval_epoch(
			model, embed_nn_list, recon_nn_list,
			validation_data, recon_adj_list, optimizer,
			batch_size, device)
		print('  - (Eval): {loss_regress: 7.4f}, '
		      ' recon: {loss_recon:3.2f}, pearson: {corr1:3.2f}, spearman: {corr2:3.2f}, '
		      ' pearson_pos: {corr1_pos:3.2f}, spearman_pos: {corr2_pos:3.2f}, '
		      ' roc: {roc:3.2f}, aupr: {aupr:3.2f}, '
		      'elapse: {elapse:3.2f} s'.format(
			loss_regress=valid_loss_regress,
			loss_recon=valid_loss_recon,
			corr1=valid_corr1,
			corr2=valid_corr2,
			corr1_pos=valid_corr1_pos,
			corr2_pos=valid_corr2_pos,
			roc=valid_roc,
			aupr=valid_aupr,
			elapse=(time.time() - start)))
		
		valid_metric += [valid_corr1]
		
		start = time.time()
		test_loss_regress, test_loss_recon, test_corr1, test_corr2, test_corr1_pos, test_corr2_pos, test_roc, test_aupr = eval_epoch(
			model, embed_nn_list, recon_nn_list,
			test_data, recon_adj_list,
			optimizer,
			batch_size, device)
		print('  - (Test): {loss_regress: 7.4f}, '
		      ' recon: {loss_recon:3.2f}, pearson: {corr1:3.2f}, spearman: {corr2:3.2f}, '
		      ' pearson_pos: {corr1_pos:3.2f}, spearman_pos: {corr2_pos:3.2f}, '
		      ' roc: {roc:3.2f}, aupr: {aupr:3.2f}, '
		      'elapse: {elapse:3.2f} s'.format(
			loss_regress=test_loss_regress,
			loss_recon=test_loss_recon,
			corr1=test_corr1,
			corr2=test_corr2,
			corr1_pos=test_corr1_pos,
			corr2_pos=test_corr2_pos,
			roc=test_roc,
			aupr=test_aupr,
			elapse=(time.time() - start)))
		
		checkpoint = {
			'model_link': model.state_dict(),
			'epoch': epoch_i}
		
		# print (valid_corr2, max(valid_accus))
		
		if valid_corr1 >= max(valid_metric):
			print("%.2f to %.2f saving" % (valid_corr1, float(max(valid_metric))))
			torch.save(checkpoint, save_path)
		
		if len(checkpoint_list) < ensemble_num:
			checkpoint_list.append(deepcopy(checkpoint))
			checkpoint_score.append(valid_corr1)
		else:
			if valid_corr1 >= min(checkpoint_score):
				checkpoint_list[np.argmin(checkpoint_score)] = deepcopy(checkpoint)
				checkpoint_score[np.argmin(checkpoint_score)] = valid_corr1
				no_change = 0
			else:
				no_change += 1
		if no_change >= 10:
			break
		
		# torch.cuda.empty_cache()
		
		# print(model.encode1.mul_head_attn.alpha_static, model.encode1.mul_head_attn.alpha_dynamic,
		# 	  model.encode1.pff_n2.alpha, model.encode1.pff_n1.alpha)
		print("checkpoint scores", checkpoint_score, "no change", no_change)
	
	checkpoint_list = np.array(checkpoint_list)
	checkpoint_score = np.array(checkpoint_score)
	checkpoint_list = checkpoint_list[np.argsort(checkpoint_score)[::-1]]
	return checkpoint_list, checkpoint_score


# checkpoint = torch.load(save_path)


def pre_train(embed_nn_list, recon_nn_list, recon_adj_list, optimizer, epochs, batch_size, device):
	x = torch.arange(1, gene_num, dtype=torch.long, device=device)
	
	for epoch_i in trange(epochs):
		start = time.time()
		# print('[ Epoch', epoch_i, 'of', epochs, ']')
		index = torch.randperm(len(x))
		x = x[index]
		batch_num = int(math.floor(len(x) / batch_size))
		# bar = trange(batch_num, desc=' - (Train) ', leave=False, )
		bar = range(batch_num)
		total_loss_recon = 0
		for i in bar:
			batch_data = x[i * batch_size:(i + 1) * batch_size]
			loss_recon = 0
			
			for j, embed_nn, recon_nn, recon_adj in enumerate(zip(embed_nn_list, recon_nn_list, recon_adj_list)):
				loss_recon += forward_batch_recon(embed_nn, recon_nn, batch_data, recon_adj, lambda_list[j])
			
			loss = loss_recon
			
			optimizer.zero_grad()
			
			# backward
			loss.backward()
			
			# update parameters
			optimizer.step()
			
			total_loss_recon += loss_recon.item()
	

def one_training_procedure(train_data, valid_data, test_data, gene_num):
	# graphsage_embedding, recon_nn, node_embedding, hypersagnn = get_baseline_model(gene_num, embed_dim, auxi_m, auxi_adj)
	graphsage_embedding, recon_nn, node_embedding, hypersagnn = get_model(gene_num, embed_dim, auxi_m, auxi_adj)
	
	parameter_list = list(hypersagnn.parameters())
	for nn in recon_nn:
		parameter_list += list(nn.parameters())
	optimizer = torch.optim.Adam(parameter_list, lr=1e-3)
	if isinstance(hypersagnn, Hyper_SAGNN):
		pre_train(embed_nn_list=graphsage_embedding,
		          recon_nn_list=recon_nn,
		          recon_adj_list=auxi_adj,
		          optimizer=optimizer,
		          epochs=40,
		          batch_size=96,
		          device=device)
	
	optimizer = torch.optim.Adam(parameter_list, lr=1e-3)
	
	
	
	ckpt_list, score_list = train(model=hypersagnn,
	                              embed_nn_list=graphsage_embedding,
	                              recon_nn_list=recon_nn,
	                              training_data=train_data,
	                              validation_data=valid_data,
	                              test_data=test_data,
	                              recon_adj_list=auxi_adj,
	                              optimizer=optimizer,
	                              epochs=60,
	                              batch_size=96,
	                              device=device)
		
	ensemble = 0
	count = 0
	for checkpoint in ckpt_list:
		hypersagnn.load_state_dict(checkpoint['model_link'])
		
		start = time.time()
		test_loss_regress, test_loss_recon, test_corr1, test_corr2, \
		test_corr1_pos, test_corr2_pos, test_roc, test_aupr = eval_epoch(hypersagnn, graphsage_embedding, recon_nn,
		                                                                 test_data, auxi_adj,
		                                                                 optimizer,
		                                                                 96,
		                                                                 device)
		print('  - (Test) regress: {loss_regress: 7.4f}, '
		      ' recon: {loss_recon:3.2f}, pearson: {corr1:3.2f}, spearman: {corr2:3.2f}, '
		      ' pearson_pos: {corr1_pos:3.2f}, spearman_pos: {corr2_pos:3.2f}, '
		      ' roc: {roc:3.2f}, aupr: {aupr:3.2f}, '
		      'elapse: {elapse:3.2f} s'.format(
			loss_regress=test_loss_regress,
			loss_recon=test_loss_recon,
			corr1=test_corr1,
			corr2=test_corr2,
			corr1_pos=test_corr1_pos,
			corr2_pos=test_corr2_pos,
			roc=test_roc,
			aupr=test_aupr,
			elapse=(time.time() - start)))
		pred = hypersagnn.predict(test_data[0])
		if isinstance(hypersagnn, Hyper_SAGNN):
			ensemble += pred
		else:
			ensemble = pred
		count += 1
	
	return ckpt_list, score_list, hypersagnn, graphsage_embedding, recon_nn, optimizer, ensemble / count


if __name__ == '__main__':
	try:
		if torch.cuda.is_available():
			current_device = get_free_gpu()
		else:
			current_device = 'cpu'
			
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print (sys.argv)
		positive_thres, gene_num, embed_dim, split_loc, save_dir, save_string = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),sys.argv[4], sys.argv[5], sys.argv[6]
		save_path = "../Temp/%s/model_%s" %(save_dir, save_string)
		
		main_task_loss_func = [log_cosh]
		reconstruct_loss_func = sparse_mse
		
		tuples = np.load("../data/tuples.npy").astype('int')
		y = np.load("../data/y.npy").astype('float32')
		significance = np.load("../data/sign.npy").astype('float32')
		# We use the negative log10 significance score as the sample weight
		significance = -np.log10(significance + 1e-15)
		
		# We take the negative of y, because all of the y are negative,
		# by doing so, we could directly use the prediction from regression to calculate aupr without multiplies it with -1
		y = -y
		print("num y that pases thres", np.sum(np.abs(y) > positive_thres), np.sum(y > positive_thres),
		      np.sum(-y > positive_thres), len(y))
		print("tuples, min, max", tuples, np.min(tuples), np.max(tuples))
		print("min, max of y", np.min(y), np.max(y))
		scalar = StandardScaler().fit(y.reshape((-1, 1)))
		y = scalar.transform(y.reshape((-1, 1))).reshape((-1))
		
		
		# Get the training/valid/testing indexes
		indexs = np.load(split_loc, allow_pickle=True)
		train_index, valid_index, test_index = indexs[0], indexs[1], indexs[2]
		
		train_data = tuples[train_index]
		train_y = y[train_index]
		train_sign = significance[train_index]
		valid_data = tuples[valid_index]
		valid_y = y[valid_index]
		valid_sign = significance[valid_index]
		test_data = tuples[test_index]
		test_y = y[test_index]
		test_sign = significance[test_index]
		
		print("train/valid/test data shape", train_data.shape, valid_data.shape, test_data.shape)
		
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
		
		
		# baseline(train_data, train_y, test_data, test_y)
		
		train_data = torch.from_numpy(train_data).to(device)
		valid_data = torch.from_numpy(valid_data).to(device)
		test_data = torch.from_numpy(test_data).to(device)
		
		train_y = torch.from_numpy(train_y).to(device)
		valid_y = torch.from_numpy(valid_y).to(device)
		test_y = torch.from_numpy(test_y).to(device)
		
		train_sign = torch.from_numpy(train_sign).to(device)
		valid_sign = torch.from_numpy(valid_sign).to(device)
		test_sign = torch.from_numpy(test_sign).to(device)
		
		ckpt_list, score_list, hypersagnn, graphsage_embedding, recon_nn, optimizer, ensemble_part = one_training_procedure(
			(train_data, train_y, train_sign),
			(valid_data, valid_y, valid_sign),
			(test_data, test_y, test_sign), gene_num)
		
		torch.save(ckpt_list, "../Temp/%s/%s_model_list" % (save_dir, save_string))
		np.save("../Temp/%s/ensemble_%s.npy" %(save_dir, save_string), ensemble_part.detach().cpu().numpy())
		print ("finish saving")
	except Exception as e:
		print ("training error", e)
		raise e