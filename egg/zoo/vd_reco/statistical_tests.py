import pandas as pd
import glob
import json
import numpy as np
import os 
import statsmodels.api as sm
from scipy.stats import spearmanr
import argparse

from egg.zoo.vd_reco.utils_analyze import (
    listdict2dictlist, parse_log, common_filter_runs,
)

def retrieve_results(directories):
    """ Glob through directories, parse and return results as a dataframe.
    convention: 
    - the logs are in a text file with a hash name H,
    - the corresponding experiments directories are in H_I
    """
    set_fn = set()
    data = []
    data_var = []
    fns = [glob.glob(os.path.join(d, '*')) for d in directories]
    fns = sum(fns, []) # flatten list of list
    # fns contain subdirectories, log files and .json specifying cfg
    for fn in fns:
        if fn[-2:] == '_I' or fn[-5:] == '.json':
            continue
        print("Read", fn)
        if fn in set_fn:
            print("Already parsed", fn)
            continue
        set_fn.add(fn)
        try:
            cfg, tr, te = parse_log(fn)
        except Exception as e:
            print("ERR!!!", e)
            continue
        # find best test loss:
        indices = range(len(te['loss']))
        tuples = zip(indices, te['loss'])
        sorted_ = sorted(tuples, key= lambda e: e[1])
        best_idx = sorted_[0][0] 
        # compute corresponding best train loss
        # but the validation is less frequent than the train
        matching_train_idx = best_idx * int(cfg['validation_freq'])
        final_train_loss = tr['loss_objs'][matching_train_idx]
        cfg.update({
            'final_length':  te['length'][best_idx],
            'final_loss': te['loss_objs'][best_idx],
            'final_train_loss': final_train_loss,
            'final_H1': te['entropy_1.0'][best_idx],
            'final_H2': te['entropy_2.0'][best_idx],
            'final_H3': te['entropy_3.0'][best_idx],
        })
        for k in te.keys():
            if k.startswith('length'):
                cfg.update({
                    'init_' + k: te[k][0],
                    'final_' + k: te[k][best_idx],
                })
                
        try:
            def collapse_edges(edges):
                return '-'.join(sorted(list(edges)))
            def opposite_edge(edge):
                """ given "2,1" returns "1,2"
                """
                split_edge = edge.split(',')
                return split_edge[1] + ',' + split_edge[0]

            def select_edges_from_diffs(diffs):
                """ diffs: Counter of edges like "2,1"
                Returns the list of most frequent edges for each pairs
                """
                edges = []
                edges_counts = [(k,v) for k, v in diffs.items()]
                sorted_edges_counts = sorted(edges_counts, key=lambda e: e[1], reverse=True)
                for k, v in sorted_edges_counts:
                    if opposite_edge(k) not in edges:
                        edges.append(k)
                return edges

            compo_json = os.path.join(fn + '_I', 'concatenability_test.json')
            with open(compo_json, 'r') as f:
                D = json.load(f)
                for k, v in D['loss_diffs'].items():
                    cfg['ld_' + k] = v
                for k, v in D['log_prob_diffs'].items():
                    cfg['lP_' + k] = v
                for k, v in D['loss_best_diffs'].items():
                    cfg['concatL_' + k] = v
                #  for k, v in D['loss_best_diffs_trunc'].items():
                #      cfg['concatL_trunc_' + k] = v
                for k, v in D['log_prob_best_diffs'].items():
                    cfg['concatS_' + k] = v
                for k, v in D['loss_sm_diffs'].items():
                    cfg['lC_' + k] = v
                for k, v in D['log_prob_sm_diffs'].items():
                    cfg['lPC_' + k] = v
                # Old transitivity scores (based on graph theory) have been
                # removed. The preferences remain:
                cfg['trans_L_edges'] = collapse_edges(
                    select_edges_from_diffs(D['transitivity_listener']['diffs'])
                )
                cfg['trans_S_edges'] = collapse_edges(
                    select_edges_from_diffs(D['transitivity_speaker']['diffs'])
                )
                # These are essentially "new" transitivity scores, much simpler
                # and intuitive:
                cfg['loss_gap'] = D['loss_b_diffs']['mean']
                cfg['log_prob_gap'] = D['log_prob_b_diffs']['mean']
                #  cfg['loss_gap_norm'] = D['loss_b_diffs_norm']['mean']
                #  cfg['log_prob_gap_norm'] = D['log_prob_b_diffs_norm']['mean']
                cfg['fraction_empty'] = D['fraction_empty']
    #         print(cfg, D.keys())
        except Exception as e:
            pass
        try:
            per_arg_H_msg_json = os.path.join(fn + '_I', 'per_arg_H_msg.json')
            with open(per_arg_H_msg_json, 'r') as f:
                D = json.load(f)
                cfg['avg_H'] = float(D['freq_weighted_avg_H'])
        except Exception as e:
            pass

        try:
            context_json = os.path.join(fn + '_I', 'context_ind_train.json')
            with open(context_json, 'r') as f:
                D = json.load(f)
                cfg['CI'] = max(0, float(D['CI']))

        except Exception as e:
            pass

        try:
            role_pred_json = os.path.join(fn + '_I', 'role_predict.json')
            with open(role_pred_json, 'r') as f:
                d = json.load(f)
                cfg['role_test_tfm'] = d['tfm']['test_loss']
                cfg['role_test_1gram'] = d['1gram']['test_loss']
                cfg['role_test_2gram'] = d['gram2']['test_loss']
        except Exception as e:
            print(f"Couldn't read {role_pred_json}")
            print(e)
        try:
            generalisation_json = os.path.join(fn + '_I', 'test_generalization.json')
            with open(generalisation_json, 'r') as f:
                D = json.load(f)
                cfg['test_generalization'] = D['loss_reco'] 
                cfg['test_generalization1'] = D['loss_reco_1'] 
                cfg['test_generalization2'] = D['loss_reco_2'] 
                cfg['test_generalization3'] = D['loss_reco_3'] 
        except Exception as e:
            print(f"Couldn't read test_generalization.json")
            print(e)
        try:
            generalisation_json = os.path.join(fn + '_I',
                    'test_generalization_iid.json')
            with open(generalisation_json, 'r') as f:
                D = json.load(f)
                cfg['test_generalization_iid'] = D['loss_reco'] 
                cfg['test_generalization_iid1'] = D['loss_reco_1'] 
                cfg['test_generalization_iid2'] = D['loss_reco_2'] 
                cfg['test_generalization_iid3'] = D['loss_reco_3'] 
        except Exception as e:
            print(f"Couldn't read test_generalization_iid.json")
            print(e)
        data_var.append(cfg)
    df = pd.DataFrame(data_var)

    # a bit of filtering:
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore', downcast='float'))
    df['flat_attention'] = df['flat_attention'].fillna('False')
    df['diff_lengths'] = df['final_length_3.0'] - df['final_length_1.0']

    def weighted_sum(prefix):
        S = 0
        n = 0
        for i in ['0,1', '0,2', '1,2']:
            S += df[prefix + '_' + i + '_mean'] * df[prefix + '_' + i + '_n']
            n += df[prefix + '_' + i + '_n']
        return S / n

    df['concatL_add'] = weighted_sum('concatL') 
    df['concatS_add'] = weighted_sum('concatS')
    #  df['concatL_trunc_add'] = weighted_sum('concatL_trunc')
    df['lC_add'] = weighted_sum('lC')
    df['lPC_add'] = weighted_sum('lPC')
    df['loss_gap_norm'] = df['loss_gap'] / df['final_loss']
    df['log_prob_gap_norm'] = df['log_prob_gap'] / df['final_H2']
    return df


def statistical_analysis(df, predicted, indep_variables):
    """ Perform linear regression, statistical sig tests and plot residuals.
    """
    assert(indep_variables[-1] == 'object_centric_dummy')  # ugly but will do, cf return
    data = df[(~ df[predicted].isna())].copy()  # remove NA
    data['object_centric_dummy'] = (data['flat_attention'] == 'False').astype(float)

    X = data[indep_variables].values
    X = sm.add_constant(X)
    y = data[predicted].values
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HC3')  # HC3 is robust to heteroscedasticity
    # I checked the residuals, and indeed, they are heterosc.! see plot:
    print(res.summary())
    print("Residual variance", np.sqrt(res.mse_resid))
    return {'coef': res.params[-1], 'pvalue': res.pvalues[-1]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs='+',
            help='All directories containing subdirectories with experiment results')
    args = parser.parse_args()
    df = retrieve_results(args.exp_dirs)
    df = common_filter_runs(df)

    print("average length differences:")
    print("length2 - length1", (df['final_length_2.0'] - df['final_length_1.0']).mean())
    print("length3 - length2", (df['final_length_3.0'] - df['final_length_2.0']).mean())
    same_edges = df[df['trans_L_edges'] == df['trans_S_edges']]
    print(f"There are {len(same_edges)} runs with consistent transitivity out of {len(df)}")
    def filter_above_median(dataframe, fields):
        filters = [(dataframe[f] > dataframe[f].quantile(0.5)) for f in fields]
        final_filter = filters[0]
        for f in filters[1:]:
            final_filter = final_filter & f
        return dataframe[final_filter]
    same_edges_high_concat = filter_above_median(same_edges, ['concatL_add', 'concatS_add'])
    print("Flat attention counts among these:")
    print(same_edges['flat_attention'].value_counts())
    print("Counts:")
    print(same_edges[['trans_L_edges', 'flat_attention']].value_counts())
    print(same_edges[['trans_L_edges']].value_counts())
    results = []

    metrics = [  # WARN don't move them around... 
        (r'$C^L \uparrow$', 'concatL_add'),
        (r'$C^S \uparrow$', 'concatS_add'),
        (r'CI', 'CI'),
        (r'$RPE \downarrow$', 'role_test_2gram'),
        (r'Ø(%)', 'fraction_empty'),
        (r'$T^L \uparrow$', 'loss_gap'),
        (r'$T^S \uparrow$', 'log_prob_gap'),
        (r'IID gene (Σα=1)', 'test_generalization_iid1'),
        (r'IID gene (Σα=2)', 'test_generalization_iid2'),
        (r'IID gene (Σα=3)', 'test_generalization_iid3'),
        (r'OoD gene (Σα=1)', 'test_generalization1'),
        (r'OoD gene (Σα=2)', 'test_generalization2'),
        (r'OoD gene (Σα=3)', 'test_generalization3'),
        #  (r'IID train reco', 'final_train_loss'),
        #  (r'IID valid reco', 'final_loss'),
        #  (r'IID test reco', 'test_generalization_iid'),
    ]

    all_H_and_lengths = ['final_H1', 'final_H2', 'final_H3',
            'final_length_1.0', 'final_length_2.0', 'final_length_3.0']
    #  Hs = ['final_H1', 'final_H2', 'final_H3']
    Hs = ['final_H1', 'final_H2', 'final_H3']
    controls = {
        #  'concatL_add': ['test_generalization_iid'],
        #  'loss_gap': ['test_generalization_iid'],
        'concatL_add': ['final_H1', 'final_H2'],
        'loss_gap': ['final_H1', 'final_H2'],
        'concatS_add': ['final_H1', 'final_H2'],
        'log_prob_gap': ['final_H1', 'final_H2'],
        'CI': ['test_generalization_iid'],  # unused, outdated?
        'role_test_2gram': ['final_H1'],
        'fraction_empty': ['test_generalization_iid'],  # unused, outdated?
        'test_generalization_iid1': ['final_H1'],
        'test_generalization_iid2': ['final_H2'],
        'test_generalization_iid3': ['final_H3'],
        'test_generalization1': ['final_H1'],
        'test_generalization2': ['final_H2'],
        'test_generalization3': ['final_H3'],
    }

    metric_internal_names = [iname for _, iname in metrics]
    metric_names = [ename for ename, _ in metrics]

    def process(res, name):
        res['metric'] = name
        if res['pvalue'] < 0.001:
            res['pvalue_star'] = '***'
        elif res['pvalue'] < 0.01:
            res['pvalue_star'] = '**'
        elif res['pvalue'] < 0.05:
            res['pvalue_star'] = '*'
        else:
            res['pvalue_star'] = ''
        res['coef_round_str'] = format(f"{res['coef']:.2g}")
        res['coef_star'] = ('$' + res['coef_round_str'] + '^{' +
                            res['pvalue_star'] + '}$')
        return res

    results_OC = {}
    results_FA = {}
    agged = {}

    def aggregate(dataframe):
        return dataframe.groupby('flat_attention').agg(['mean', 'std'])

    for name, internal_name in metrics:
        print(f"------- {name}")
        # Transitivity metrics are analyzed on a subset. Transitivity makes
        # sense only when concatenability is low enough, so we work on the
        # top 50% most concatenable runs only.
        if internal_name == r'loss_gap':
            df_to_analyze = filter_above_median(df, ['concatL_add'])
        elif internal_name == r'log_prob_gap':
            df_to_analyze = filter_above_median(df, ['concatS_add'])
        else:
            df_to_analyze = df
        # pvalues are computed for the LAST covariate which should be the target!
        covariates = controls[internal_name] + ['object_centric_dummy']
        res = statistical_analysis(df_to_analyze, internal_name, covariates)
        res = process(res, name)
        results.append(res)
        agged[internal_name] = aggregate(df_to_analyze[['flat_attention', internal_name]])[internal_name]

    results = listdict2dictlist(results)

    def mean_std_latex(mean, std, stars=''):
        return f"${mean:.2g}\pm{std:.2g}^" + "{" + stars + "}$"

    for name, internal_name, statsig in zip(metric_names,
            metric_internal_names, results['pvalue_star']):
        OC_mean = agged[internal_name]['mean']['False']
        FA_mean = agged[internal_name]['mean']['True']
        OC_std = agged[internal_name]['std']['False']
        FA_std = agged[internal_name]['std']['True']
        results_FA[name] = mean_std_latex(FA_mean, FA_std)
        results_OC[name] = mean_std_latex(OC_mean, OC_std, statsig)

    def print_subset(keys_id):
        keys = [metric_names[i] for i in keys_id]
        subset_res_FA = [results_FA[k] for k in keys]
        subset_res_OC = [results_OC[k] for k in keys]
        print('  & ' + ' & '.join(keys))
        print(' FA &' + ' & '.join(subset_res_FA))
        print(' OC &' + ' & '.join(subset_res_OC))

    print_subset((0,1))
    print_subset((5,6,3))
    print_subset((7,8,9))
    print_subset((10,11,12))
    #  print_subset((13,14,15))
