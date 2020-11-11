# DANGO: Towards the prediction of higher-order genetic interactions
This is the implementation of the algorithm DANGO for the prediction of higher-order genetic interactions.
This repo contains the code and data for the analysis of the yeast trigenic interactions studied in the manuscript *Towards the prediction of higher-order genetic interactions*.

## Requirements
- h5py
- numpy
- pytorch
- scikit-learn
- tqdm
- xgboost (used in the baseline, if you don't plan to run the baseline, it's not required)

## Data used in this repo

All the data are stored in the `./data` folder. 
- Within the `./data/yeast_network` folder, there are six yeast protein-protein interaction (PPI) networks obtained from STRING v9.1 database (http://string91.embl.de/)
- The `./data/AdditionalDataS1.tsv` is obtained from the paper *Systematic analysis of complex genetic interactions* (Kuzmin et al., Science, 2018), which contains the measured trigenic interaction scores of around 100,000 triplets

## Configure the parameters
There are several parameters when running the main program `python main.py`
The parameters and their meanings are
- `-t, --thread, default=8`, the number of parallel threads for training the ensemble of Dango models. The code is optimized for utilizing multiple GPUs. Each Dango model roughly takes about 2G GPU memory, so one 2080Ti card could train three Dango models at a time. Change this parameter based on your machine condition.
- `-m, --mode, default='train'`, can be `eval`, `train`, `predict` or combinations of them separated by `;` (no space). For instance `train;test`. 
  - When in `eval` mode, the main program would run cross-validation (either random split, gene split 1 or gene split 2). See details in the below sections
  - When in `train` mode, the main program would train a Dango model using the whole dataset
  - When in `predict` mode, the main program would use a trained Dango model to make predictions on a specific set of triplets. See details in the below sections
- `-i, --identifier, default=None`, the identifier for this run. A corresponding folder would be created under the ./temp folder where the trained model and prediction results would be stored. It can be useful when you train one model and want to use that specific model for making predictions later. When left empty, the program would automatically create an identifier with the date, time information embedded. When there is an existing folder under `./temp` with the same name as the identifier, the original one would be overwritten.
- `-s, --split, default=0`, how to split the training/validation/test set during evaluation. This parameter would only be used when `eval` is included in the mode
  - `0` stands for random split
  - `1` stands for gene split 1
  - `2` stands for gene split 2
- `-p, --predict, default=0`, which set of triplets to predict on. 
  - `0` stands for predicting on the original set of triplets (set 1 in the paper)
  - `1` stands for predicting on the set of triplets where all three genes are observed in the original dataset (set 2 in the paper)
  - `2` stands for predicting on the set of triplets where two of the genes are observed and the third gene is unobserved in the original dataset

## Usage
1. `cd Code`
2. Run `python process.py` This script would process the data in the `./data` folder. Specifically, it would:
	1. Generate two dictionaries : `./data/gene2id.npy` and `./data/id2gene.npy` which is two dictionaries that map between the gene name and node id (starts from 1). Can be load with `np.load(xx, allow_pickle=True).item()`
	2. Process PPI networks.
	3. Parse the trigenic interaction scores. The processed signals are stored as : `./data/tuples.npy`, `./data/y.npy`, `./data/sign.npy` which corresponds to the triplets, measured trigenic interaction scores and p-values respectively.
3. Run `python main.py -t 4 -i dango_run1 -m train;test -p 1`. The parameters can be configured as described in the above section. A folder named as the `--identifier` parameter would be created under the `./temp/` dir where all results would be stored. The output under that folder includes:
	1. If the `--mode` parameter includes `eval`. The training process and the final performance would be printed on the screen. The performance would also be written to a file named `result.txt`.
	2. If the `--mode` parameter includes `eval` or `train`. The selected groups of the model parameters (model enesembling trick, so there will be multiple sets of parameters) is stored as the file `Experiment_model_list`. The parameters can be transformed into a list of Dango models by
	```{python}
	from main import create_dango_list
	ckpt_list = list(torch.load("../Temp/%s/Experiment_model_list" % (dir_name), map_location='cpu'))
	dango_list = create_dango_list(ckpt_list)
	```
	3. If the model runs in the random 5-fold CV (`--mode=eval`, `--split=0`), the predicted results for each fold would be stored in an hdf5 file named as `Experiment.hdf5`. 	The structure of this hdf5 file is listed. The `ensemble_inverse` and `test_y_inverse` correspond to the predicted scores and measured scores. They are both inversely transformed to the original value scale of the measured trigenic interaction scores.
	```
	./Cross_valid_0
		./ensemble_inverse
		./test_y_inverse
	./Cross_valid_1
	```
	4. If the `--mode` parameter includes `predict`, the corresponding results would be stored as `XX_tuples.npy`, `XX_y.npy`. Only tuples with predicted signals larger than 0.05 are kept to save space. Based on the `--predict` parameter, `XX`is named as `re_eval`, `within_seen`, `two_seen_one_unseen` respectively.
	
## Cite

