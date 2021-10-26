import pandas as pd
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from statistical_tests import retrieve_results

A6_landscape = (5+7/8, 4+1/8)

def weighted_sum(prefix):
    S = 0
    n = 0
    for i in ['0,1', '0,2', '1,2']:
        S += df[prefix + '_' + i + '_mean'] * df[prefix + '_' + i + '_n']
        n += df[prefix + '_' + i + '_n']
    return S / n

def plot_transitivity_vs_role_prediction(df, fn):
    """ Show and save figure.
    """
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), 
        f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}")
    plt.figure(figsize=A6_landscape)
    transitivity_metric = 'transitivity_L'
    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data[transitivity_metric],
            data['role_test_1gram'],
            c=data['final_loss'], 
            marker=marker, label=label, s=30,
        )
    plt.ylabel(r'$RPE$')
    plt.xlabel(r'Transitivity $T^L$')

    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()

def plot_concatenability(df, fn):
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), 
        f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}")
    plt.figure(figsize=A6_landscape)

    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data['lPB_add'],
            data['lB_add'],
            c=data['final_loss'],
            marker=marker, label=label, 
        )
    plt.xlim(right=5)  
    plt.ylim(top=0)    
    plt.xlabel(r'$C^S$')
    plt.ylabel(r'$C^L$')
    plt.colorbar()
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()

if __name__ == '__main__':
    exp_dirs = ['res_proto_1B_adam', 'res_proto_1B2_adam', 'res_proto_1B3_adam']
    df = retrieve_results(exp_dirs)
    # a bit of filtering:
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore', downcast='float'))
    df['flat_attention'] = df['flat_attention'].fillna('False')
    df['lB_add'] = weighted_sum(prefix='lB')
    df['lC_add'] = weighted_sum('lC')
    df['lPB_add'] = weighted_sum('lPB')

    n_runs = len(df)
    collapsed_cond = df['final_length'] <= 1.0
    collapsed_length = df[collapsed_cond]
    print(f"{len(collapsed_length)} / {n_runs} send empty messages")
    df = df[~collapsed_cond]

    A5_landscape = (10+6/8, 8+2/8)
    fn_out = '/media/Docs/writings/grammar_emergence/fig/x_T_y_RPE_c_loss.pdf'
    plot_transitivity_vs_role_prediction(df, fn_out)
    fn_out = '/media/Docs/writings/grammar_emergence/fig/x_dlogp_y_dloss_c_loss.pdf'
    plot_concatenability(df, fn_out)
