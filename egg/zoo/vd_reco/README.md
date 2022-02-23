### Training

TODO

In my case, to each experiment corresponds a subdirectory in `res_proto_min1_adam`, `res_proto_min2_adam` and `res_proto_min2B_adam`.

### Computation of metrics & qualitative analysis

- Open `egg/zoo/vd_reco/compute_metrics.sh`. 
- At the very beginning of the file, a few binary variables allow you to select what metric to compute. For example, if you set `CONCATENABILITY=1`, concatenability and transitivity metrics will be computed. For qualitative analysis, the analysis script will output a lot of message samples with a particular syntax to indicate masks, entities, etc.
- Each analysis script put the results in the experiment directory, alongside with the weights of the best model, etc.
- Right after these variables, you will find a loop iterating over all subdirs. Please customize this. For example, loop using `for N in res_proto_min*_adam/*`.

Once you have set your binary variables and specified the paths of subdirs to analyze, run:

```
./egg/zoo/vd_reco/compute_metrics.sh
```
Go to the root (where result subdirs are)

### Analysis: tables & plots

At this step, you need trained agents and you need to have computed the various metrics. Once again, I assume results are in subdirectories of various directories that follow the pattern `res_proto_min*`

To produce latex tables, 

```
python -m egg.zoo.vd_reco.statistical_tests res_proto_min*
```

To produce plots,

```
mkdir plotdir
python -m egg.zoo.vd_reco.prepare_plots plotdir res_proto_min*
```

