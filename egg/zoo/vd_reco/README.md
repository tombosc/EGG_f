### Training

TODO

### Computation of metrics

TODO

### Analysis

Once the agents are trained and various metrics are computed, we can run `statistical_tests.py`. It pools all the experiments from the subdirectories and:

- generates latex tables,
- give various statistics like average sentence length, number of filtered runs, etc. 

It takes the list of directories as command line arguments. For instance, I have results stored in `res_proto_min1`, `res_proto_min2` and `res_proto_min2B`, so I run:

```
python -m egg.zoo.vd_reco.statistical_tests res_proto_min*
```
