#!/usr/bin/env python
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
sns.set(color_codes=True)

def main(args):

    db = sqlite3.connect(args.input)
    df = pd.read_sql("select * from estimated_ctf_parameters;", db)

    outname = os.path.splitext(os.path.basename((args.input)))[0]

    figs = []

    figs.append(ctf_res_histogram(df, args.ctf_id))
    figs.append(defocus_histogram(df, args.ctf_id))
    figs.append(ctf_res_timecourse(df, args.ctf_id))
    figs.append(plot_astigmatsm_histogram(df, args.ctf_id))
    figs.append(ctf_score_histogram(df, args.ctf_id))

    with PdfPages('%s_CTF_statistics.pdf' %outname) as pdf:
        for i in figs:
            pdf.savefig(i)


def ctf_score_histogram(df, ctf_id):
    max_ctf_score = 1.0
    min_ctf_score = 0.0

    df = df[df['SCORE'] < max_ctf_score]
    df = df[df['CTF_ESTIMATION_JOB_ID'] == ctf_id]
    score_mean = np.mean(df['SCORE'])

    fig, ax = plt.subplots(1, 1)
    sns.distplot(df['SCORE'], kde=False, ax=ax)
    #     ax.plot(df['DETECTED_RING_RESOLUTION'])
    plt.axvline(x=score_mean)
    plt.xlim(min_ctf_score, max_ctf_score)
    plt.ylabel('micrographs')
    plt.title('Distribution of CTF scores for ctf_id %s, mean = %3.2f ' % (ctf_id, score_mean))
    #     plt.savefig('ctf_res_histogram_CTF_ctf_id_%s' %ctf_id)
    return fig

def ctf_res_histogram(df, ctf_id):
    max_ctf_res = 10
    min_ctf_res = 2

    df = df[df['DETECTED_RING_RESOLUTION'] < max_ctf_res]
    df = df[df['CTF_ESTIMATION_JOB_ID'] == ctf_id]
    res_mean = np.mean(df['DETECTED_RING_RESOLUTION'])

    fig, ax = plt.subplots(1, 1)
    sns.distplot(df['DETECTED_RING_RESOLUTION'], kde=False, ax=ax)
    #     ax.plot(df['DETECTED_RING_RESOLUTION'])
    plt.axvline(x=res_mean)
    plt.xlim(min_ctf_res, max_ctf_res)
    plt.ylabel('micrographs')
    plt.title('Distribution of CTF resolution for ctf_id %s, mean = %3.2f angstroms' % (ctf_id, res_mean))
    #     plt.savefig('ctf_res_histogram_CTF_ctf_id_%s' %ctf_id)
    return fig



def defocus_histogram(df, ctf_id):
    min_defocus = 5000  ##remove if input parameter
    max_defocus = 35000  ##remove if input parameter
    max_ctf_res = 10

    df = df[df['DETECTED_RING_RESOLUTION'] < max_ctf_res]
    df = df[df['CTF_ESTIMATION_JOB_ID'] == ctf_id]
    def_mean = np.mean(df['DEFOCUS1'])

    def_mean = np.mean(df['DEFOCUS1'])
    def_mean_micron = def_mean / 10000

    fig, ax = plt.subplots(1, 1)
    sns.distplot(df['DEFOCUS1'], bins=100, kde=False)
    plt.xlim(min_defocus, max_defocus)
    plt.axvline(x=def_mean)
    plt.ylabel('micrographs')
    plt.title('Distribution of defocus values for ctf_id %s, mean = -%3.2f microns' % (ctf_id, def_mean_micron))
    #     plt.savefig('defocus_histogram_CTF_ctf_id_%s' %ctf_id)
    return fig


def ctf_res_timecourse(df, ctf_id):
    max_ctf_res = 10  ##remove if input parameter
    min_ctf_res = 2  ##remove if input parameter

    df = df[df['DETECTED_RING_RESOLUTION'] < max_ctf_res]
    df = df[df['CTF_ESTIMATION_JOB_ID'] == ctf_id]
    res_mean = np.mean(df['DETECTED_RING_RESOLUTION'])

    fig, ax = plt.subplots(1, 1)
    plt.title('CTF resolution over time \n Requires that movies were imported by new')
    x = df['CTF_ESTIMATION_ID']
    y = df['DETECTED_RING_RESOLUTION']
    sns.regplot(x=x, y=y, fit_reg=False)
    plt.ylim(2, 10)
    plt.axhline(y=res_mean)
    #     plt.savefig('ctf_res_timecourse_CTF_ctf_id_%s' %ctf_id)
    return fig


def plot_astigmatsm_histogram(df, ctf_id):
    df = df[df['CTF_ESTIMATION_JOB_ID'] == ctf_id]
    astigmatism = df['DEFOCUS1'] - df['DEFOCUS2']

    fig, ax = plt.subplots(1, 1)
    sns.distplot(astigmatism, bins=100, kde=False)
    plt.ylabel('micrographs')
    plt.title('Distribution of ASTIGMATISM for ctf_id %s' % ctf_id)
    plt.xlabel('Astigmatism (A)')
    plt.xlim(0, 2000)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .db file")
    parser.add_argument("--ctf-id", help="CTF Estimation ID to be used for plotting", default=1, type=int)
    sns.set()

    sys.exit(main(parser.parse_args()))