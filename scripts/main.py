"""

"""

import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


INVALID_SUBJECT_SCORE = 10
with open('subject_filterscores.json') as f:
    subject_filterscores = json.load(f)

with open('subject_attrsinfo.json') as f:
    subject_attrsinfo = json.load(f)


def load_raw(filename):
    mat = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    return mat


def get_all_model_runs_data(mat, starti, endi):
    data = dict()
    for j in range(starti-1, endi):
        subj_run = mat['results'][j]
        data[j+1] = subj_run
    return data


def plot_q_pc(qvalues, pcvalue,  model_name, params):
    plt.plot(qvalues[:, 0], label='q-values')
    plt.plot(qvalues[:, 1:])
    plt.plot(pcvalue, '.', label='choice prob')
    plt.legend()
    plt.xlabel('trial')
    xticks = np.arange(0, 181, 30)
    for xt in xticks:
        plt.plot([xt, xt], [-1, 2], 'k:')
    plt.xticks(xticks)
    plt.title(f'{model_name}: transformed params = {params}')
    plt.show()
    return


if __name__ == '__main__':

    subject_attrs = ['age', 'sex', 'mst', 'ldi', 'pss', 'quic', 'expDate']

    # load output files
    # model = '1Sampler'
    # resultfile = '/Users/usingla/MATLAB-Drive/resultmatfiles/ctxsamplerv3/ctxsamplerv3_results'
    # (starti, endi) for resultmatfiles of type ctxsamplerv3_results{starti}_{endi}
    # files = sorted([(1, 16), (17, 58), (59, 74), (75, 86), (87, 150), (151, 251), (252, 256)])
    # columns = ['subject', 'score', 'model', 'num_params', 'll', 'bic',
    #            *['raw_alpha', 'raw_beta', 'raw_beta_p'],
    #            *['alpha', 'beta', 'beta_p'],
    #            'exitflag', 'file', 'subjectid', *subject_attrs]

    # model = 'Plain Q-learning'
    # resultfile = '/Users/usingla/MATLAB-Drive/resultmatfiles/ctxtd2/ctxtd2_results'
    # (starti, endi) for resultmatfiles of type ctxtd2_results{starti}_{endi}
    # files = sorted([(1, 256)])
    # columns = ['subject', 'score', 'model', 'num_params', 'll', 'bic',
    #            *['raw_alpha', 'raw_beta', 'raw_beta_p'],
    #            *['alpha', 'beta', 'beta_p'],
    #            'exitflag', 'file', 'subjectid' , *subject_attrs]

    model = 'Hybrid'
    resultfile = '/Users/usingla/MATLAB-Drive/resultmatfiles/ctxhybrid/ctxhybrid_results'
    # (starti, endi) for resultmatfiles of type ctxhybrid_results{starti}_{endi}
    files = sorted([(1, 14), (15, 28), (29, 88), (89, 99), (100, 107), (108, 111), (112, 192), (193, 219), (220, 227), (228, 256)])
    columns = ['subidx', 'score', 'model', 'num_params', 'll', 'bic',
               *['raw_alpha_sampler', 'raw_beta_sampler', 'raw_beta_p', 'raw_alpha_td', 'raw_beta_td'],
               *['alpha_sampler', 'beta_sampler', 'beta_p', 'alpha_td', 'beta_td',],
               'exitflag', 'file', 'subjectid', *subject_attrs]

    print(files)
    run_data = dict()
    for idxs in files:
        starti, endi = idxs
        raw_matobj = load_raw(f'{resultfile}{starti}_{endi}.mat')
        # print(raw_matobj['results'])
        run_data.update(get_all_model_runs_data(raw_matobj, starti, endi))
    print("subjects loaded", len(run_data.keys()))

    # get params
    params = []
    subject_ids = {}
    count_scores = defaultdict(int)
    for subi in run_data:
        sub = run_data[subi]
        subidx = sub.file.split('.')[2].split('_')[1]
        subject_id = sub.file.split('_')[-1].split('.')[0]
        score = subject_filterscores.get(subject_id, INVALID_SUBJECT_SCORE)
        count_scores[score] += 1
        if score >= 3:
            print("Skipped (high filter score)", subidx, subject_id)
            continue
        assert subi == int(subidx)
        params.append([subi, score, model, sub.numParams, sub.nLogLik, sub.BIC, *sub.params, *sub.transformedParams,
                       sub.exitflag, sub.file, subject_id, *list(subject_attrsinfo[subject_id].values())])
        subject_ids[subidx] = subject_id

    print("Subjects after filter", len(subject_ids))
    print("filter score distribution", count_scores)

    # write to csv
    with open(f'{model}_params.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(params)

    # sample plot
    # sub = run_data[2]
    # plot_q_pc(sub.runQ, sub.pc, model, sub.transformedParams)
