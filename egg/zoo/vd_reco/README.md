### Training

TODO

In my case, to each experiment corresponds a subdirectory in `res_proto_min1_adam`, `res_proto_min2_adam` and `res_proto_min2B_adam`.

### Computation of metrics

- Open `egg/zoo/vd_reco/compute_metrics.sh`. 
- At the very beginning of the file, a few binary variables allow you to select what metric to compute. For example, if you set `CONCATENABILITY=1`, concatenability and transitivity metrics will be computed. Each analysis script put the results in the experiment directory, alongside with the weights of the best model, etc.
- Right after these variables, you will find a loop iterating over all subdirs. Please customize this. For example, loop using `for N in res_proto_min*_adam/*`.

From the root, you can modify and run:

```
./egg/zoo/vd_reco/compute_metrics.sh
```
Go to the root (where result subdirs are)

### Analysis

Once the agents are trained and various metrics are computed, we can run `statistical_tests.py`. It pools all the experiments from the subdirectories and:

- generates latex tables,
- give various statistics like average sentence length, number of filtered runs, etc. 

It takes the list of directories as command line arguments. For instance, I run:

```
python -m egg.zoo.vd_reco.statistical_tests res_proto_min*
```
