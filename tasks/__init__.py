
import os
import logging
import pickle
from typing import List

import numpy as np
import pandas as pd

from .downstream import DownstreamTask
from .binary import BinaryDownstreamTask
from .multi import MultiDownstreamTask


def get_task(task_name: str, task_type: str) -> DownstreamTask:
    task_name = task_name.lower()
    task_type = task_type.lower()

    if task_type == 'binary':
        task_cls = BinaryDownstreamTask
    elif task_type == 'multi':
        task_cls = MultiDownstreamTask
    else:
        raise NotImplementedError

    if task_name in {'cifar10', 'stl10'}:
        task = task_cls(task_name, task_name, n_classes=10, use_coarse_label=False)
    elif task_name == 'cifar100':
        task = task_cls(task_name, 'cifar100', n_classes=20, use_coarse_label=True)
    else:
        raise NotImplementedError
    return task


def run_downstreams(
        downstream: str,
        task_type: str,
        output_path: str,
        train_paths: List[str],
        test_paths: List[str],
        load: bool = True
) -> pd.DataFrame:

    # Load
    if load and os.path.exists(output_path):
        logging.info(f'Downstream results are loaded from {output_path}.')
        return pd.read_csv(output_path, index_col=False)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Run Downstream Tasks
    task = get_task(downstream, task_type)

    logging.info('Start running downstream task ....')
    downstream_err_list = []
    pred_prob_list = []
    target_prob_list = []
    for train_pth, test_pth in zip(train_paths, test_paths):
        with open(train_pth, 'rb') as f:
            data = pickle.load(f)
            rep_ref = data['emb']
            lab_ref = data['label']

        with open(test_pth, 'rb') as f:
            data = pickle.load(f)
            rep_test = data['emb']
            lab_test = data['label']

        task.fit(rep_ref, lab_ref)
        downstream_err, pred_prob, target_prob = task.run_test(rep_test, lab_test)

        downstream_err_list.append(downstream_err)
        pred_prob_list.append(pred_prob)
        target_prob_list.append(target_prob)

    task_idx = downstream_err_list[0]['task_idx']
    assert all(np.allclose(arr, target_prob) for arr in target_prob_list)
    assert all(np.allclose(err['task_idx'], task_idx) for err in downstream_err_list)
    logging.info('Complete performing downstream task.')

    # Compute global uncertainty
    p = np.mean(pred_prob_list, axis=0)  # (n_tasks, n_samples, n_classes)
    p_target = p[target_prob == 1].reshape(target_prob.shape[:2])
    pred_var = np.sum(np.var(pred_prob_list, axis=0), axis=-1)  # (n_tasks, n_samples)
    pred_ent = -np.sum(p * np.log2(p, out=np.zeros_like(p), where=(p != 0)), axis=-1)  # (n_tasks, n_samples)
    pred_brier = np.sum((p - target_prob)**2, axis=-1)  # (n_tasks, n_samples)

    global_errs = {
        'task_idx': task_idx,
        'pred_entropy': pred_ent,
        'brier': pred_brier,
        'pred_err': 1 - p_target,
        'log_pred_prob': np.log(p_target)
    }

    # Parse as DataFrame
    results = []
    n_tasks, n_samples = pred_var.shape
    for i in range(n_samples):
        for j in range(n_tasks):
            if np.isnan(pred_var[j, i]):
                continue

            for i_model, err_dict in enumerate([global_errs, *downstream_err_list]):
                results.append({
                    'data_idx': i,
                    'model_idx': i_model - 1,
                    **{key: value[j, i] for key, value in err_dict.items()}
                })

    # Save
    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)
    return results
