import pandas as pd
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from .statistical_tests import retrieve_results
from egg.zoo.vd_reco.utils_analyze import common_filter_runs

A6_landscape = (5+7/8, 4+1/8)

def weighted_sum(prefix):
    """ Compute a weighted sum over dataframes.

    Careful! Don't do that on metrics that do not have a min/max in there, it
    doesn't make sense.
    """
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
    assert (len(df_flat) > 0) and (len(df_oc) > 0), f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}"
    plt.figure(figsize=A6_landscape)
    transitivity_metric = 'transitivity_L'
    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data[transitivity_metric],
            data['role_test_2gram'],
            #  data['avg_H'],
            c=data['final_loss'], 
            marker=marker, label=label, s=30,
        )
    plt.xlabel(r'Transitivity $T^L$')
    plt.ylabel(r'$RPE$')
    #  plt.ylabel(r'$H[E|R]$')

    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()


def plot_gap_vs_role_prediction(df, fn, listener, norm):
    """ Show and save figure.
    """
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}"
    plt.figure(figsize=A6_landscape)
    if listener:
        gap_metric = 'loss_gap'
        gap_label = r'$T^L$'
    else:
        gap_metric = 'log_prob_gap'
        gap_label = r'$T^S$'
    if norm:
        gap_metric += '_norm'
    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data[gap_metric],
            data['role_test_2gram'],
            c=data['final_loss'], 
            marker=marker, label=label, s=30,
        )
    plt.xlabel(gap_label)
    plt.ylabel(r'$RPE$')

    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()

def plot_transitivity_vs_gap(df, fn, listener=False):
    """ Show and save figure.
    """
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}"
    plt.figure(figsize=A6_landscape)
    if listener:
        transitivity_metric = 'transitivity_L'
        transitivity_label = '$T^L$'
        gap_metric = 'loss_gap_norm'
    else:
        transitivity_metric = 'transitivity_S'
        transitivity_label = '$T^S$'
        gap_metric = 'log_prob_gap_norm'

    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data[transitivity_metric],
            data[gap_metric],
            c=data['final_loss'], 
            marker=marker, label=label, s=30,
        )
    plt.xlabel(r'Transitivity ' + transitivity_label)
    plt.ylabel(r'Gap')

    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()


def plot_concatenability(df, fn):
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}"
    plt.figure(figsize=A6_landscape)

    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data['concatS_add'],
            data['concatL_add'],
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

def plot_concatenability_vs_transitivity(df, fn):
    df_flat = df[df['flat_attention'] == 'True']
    df_oc = df[df['flat_attention'] == 'False']
    assert (len(df_flat) > 0) and (len(df_oc) > 0), f"Missing one type of run: FA:{len(df_flat)}, OC:{len(df_oc)}"
    plt.figure(figsize=A6_landscape)

    for data, marker, label in zip([df_flat, df_oc], ['x', 'o'], ['FA', 'OC']):
        ax = plt.scatter(
            data['concatL_add'],
            data['transitivity_L'],
            c=data['final_loss'],
            marker=marker, label=label, 
        )
    #  plt.xlim(right=5)
    #  plt.ylim(top=0)
    plt.xlabel(r'$C^L$')
    plt.ylabel(r'$T^L$')
    plt.colorbar()
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fn, bbox_inches='tight', pad_inches =0)
    print(f"Saved picture in: {fn}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=str)
    parser.add_argument("exp_dirs", nargs='+',
            help='All directories containing subdirectories with experiment results')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='If set, no plots are output. Interactive python.')
    args = parser.parse_args()

    df = retrieve_results(args.exp_dirs)
    df = common_filter_runs(df)
    # a bit of filtering:

    # filter out empty messages
    n_runs = len(df)
    collapsed_cond = df['final_length'] <= 1.0
    collapsed_length = df[collapsed_cond]
    print(f"{len(collapsed_length)} / {n_runs} send empty messages")
    df = df[~collapsed_cond]

    if args.interactive:
        df['sender_mask_padded_d'] = (df['sender_mask_padded'] == 'True').astype(float)
        #  CI = df['CI'].values
        #  CI_30 = df['MI_per_K_30'].values
        #  masking_on = df['sender_mask_padded_d'].values
        import pdb; pdb.set_trace()
    else:
        # show and save plots
        A5_landscape = (10+6/8, 8+2/8)
        #  fn_out = os.path.join(args.outdir, 'x_TL_y_gap.pdf')
        #  plot_transitivity_vs_gap(df, fn_out, listener=True)
        #  fn_out = os.path.join(args.outdir, 'x_TS_y_gap.pdf')
        #  plot_transitivity_vs_gap(df, fn_out, listener=False)
        #  fn_out = os.path.join(args.outdir, 'x_T_y_RPE_c_loss.pdf')
        #  plot_transitivity_vs_role_prediction(df, fn_out)
        fn_out = os.path.join(args.outdir, 'x_dlogp_y_dloss_c_loss.pdf')
        plot_concatenability(df, fn_out)
        fn_out = os.path.join(args.outdir, 'x_gapL_y_RPE.pdf')
        plot_gap_vs_role_prediction(df, fn_out, listener=True, norm=False)
        fn_out = os.path.join(args.outdir, 'x_gapS_y_RPE.pdf')
        plot_gap_vs_role_prediction(df, fn_out, listener=False, norm=False)
        fn_out = os.path.join(args.outdir, 'x_gapL_norm_finalloss_y_RPE.pdf')
        plot_gap_vs_role_prediction(df, fn_out, listener=True, norm=True)
        fn_out = os.path.join(args.outdir, 'x_gapS_norm_H2_y_RPE.pdf')
        plot_gap_vs_role_prediction(df, fn_out, listener=False, norm=True)
