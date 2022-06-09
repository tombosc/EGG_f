Code for the TACL paper (soon to appear, hopefully) "The emergence of argument structure in artificial languages" (Tom Bosc, Pascal Vincent).

You can install, train, compute metrics and analyse the results. We also provide the raw metrics (without the model weights which are quite heavy) to allow for different statistical analyses (TODO).

Please ping me via email if I don't see that you've opened an issue.

### Installation

You need the EGG dependencies + [simple_parsing](https://github.com/lebrice/SimpleParsing/). 

For reproducibility: the exact list of packages I've used to run the experiments are in [package-list.txt](package-list.txt). I've used CUDA for faster training. Make sure the versions of `cudatoolkit` and `pytorch` will work with your own GPU. Then you can install the same environment (`egg`) by using:

```
conda create -n egg --file egg/zoo/vd_reco/package-list.txt
```

The Proto-Role dataset of Reisinger & al (2015) is available [here](http://decomp.io/projects/semantic-proto-roles/protoroles_eng_pb.tar.gz) ([project page](http://decomp.io/projects/semantic-proto-roles/)). The `protoroles_eng_pb` in the archive should be extracted in your `EGG` root, so that `EGG/protoroles_eng_pb/protoroles_eng_pb_08302015.tsv` exists. 

### Training

To start training, load your environment and run the random search script:

```
conda activate egg
# run 3 random runs sequentially, with a random seed (for the search) of 0
# (the random seed of 0 is just for the hyperparameters, not for the model 
#  init or the dataset split)
./egg/zoo/vd_reco/train_wrap.sh 0 3
```

Edit `train_wrap.sh` to understand where results are stored, what are the arguments of this script, and how to select an hyperparameter config file. 

The hyperparameters are stored in [egg/zoo/vd_reco/hyperparam_grid/](hyperparam_grid/) in `json` files. Each parameter should be pretty self contained, except a few that I can explain here:

- `max_len`: maximum message length, excluding the end of sentence token.
- `patience`: patience is a parameter for early stopping: everytime we reach some minimum loss value, a count is initialised to `patience`. After each epoch this count is decreased and the training stops when there is no more patience. 
- `validation_freq`: an evaluation on the validation set is performed every `validation_freq` epochs
- `length_cost`: $\alpha$ in the paper
- `ada_len_cost_thresh`: $\tau$ in the paper
- `free_symbols`: $n_{min}$ in the paper
- `n_epoch`: the maximum number of epoch. It usually stops before, thanks to early stopping (cf `patience` and `validation_freq`)
- `random_seed` and `dataset_seed`: `dataset_seed` only controls the split/subsampling of the dataset into train, valid and test. It is the same across all experiments. `random_seed` is the seed for parameter initialisation.
- `sender_mask_padded`: whether the sender can attend over dummy entities or not. Does not seem to change much.

In the paper, I've used `proto_min1_adam.json`, `...min2...` and `...min2B...` hyperparam files. Thus the model weights and experiment results are stored in `res_proto_min1_adam`, `res_proto_min2_adam` and `res_proto_min2B_adam` (at the root of `EGG` directory).

### Computation of metrics & qualitative analysis

Once you've run some experiments, you can compute various metrics and the qualitative analysis script. 

Open `egg/zoo/vd_reco/compute_metrics.sh`. Binary variables allow you to select what metric(s) to compute. By default, only the metrics used in the camera ready of the paper are set to 1. For the qualitative analysis, the analysis script will output a lot of message samples with a particular syntax to indicate masks, entities, etc.

A main loop iterates over all subdirs. Each analysis script put their result in the experiment subdirectory.

Once you have set your binary variables and specified the paths of subdirs to analyze, run:

```
./egg/zoo/vd_reco/compute_metrics.sh
```

### Analysis: tables & plots

Once again, I assume results are in subdirectories of various directories that follow the pattern `res_proto_min*`. To produce the latex tables, run 

```
python -m egg.zoo.vd_reco.statistical_tests res_proto_min*
```

To produce plots,

```
mkdir plotdir
python -m egg.zoo.vd_reco.prepare_plots plotdir res_proto_min*
```

### Other things, unused in the paper

- Context independence: a metric of context independence, an improvement over Bogin & al 2018 proposed metric was implemented but is not documented. (TODO)
- Unordered outputs (`--unordered_classical`): a variant of the decoder, where the decoder jointly predicts a feature vector AND a role (including a dummy, not used role, if I remember correctly), and then these are matched to the ground truth using the Hungarian algorithm. (In contrast, the version presented in the paper predicts the features for each role, as well as a binary variable indicating whether there is an object with that role.) Shouldn't change much. It would be more scalable to use this version if we had many roles.
