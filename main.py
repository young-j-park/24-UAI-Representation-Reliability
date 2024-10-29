
import os
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import yaml

from tasks import run_downstreams
from evaluation.evaluate import evaluate_score
from evaluation.evaluator import ReliEvaluator
from utils import df_to_dict

warnings.filterwarnings('ignore')


def get_args_parser():
    # parse args
    parser = argparse.ArgumentParser(description='Set arguments for SSL-Uncertainty', add_help=False)

    # Model
    parser.add_argument('--config-path', required=True, type=str)

    # Paths
    parser.add_argument('--output-dir', default='./results', type=str)

    # Downstream Inference
    parser.add_argument('--seed', default=0, type=int, help='Random seed deciding the reference set')
    parser.add_argument('--task-type', type=str, default=None, choices=['binary', 'multi'],
                        help='Task type for downstream tasks. If set, it overrides the task type from the config.')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser


def main(args):
    # Logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s")

    # Load a config
    config_path = args.config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    args.ssl = model_cfg['ssl']
    args.arch = model_cfg['arch']
    args.pretrain = model_cfg['pretrain']

    # Load / Run downstream tasks
    downstream_cfg = config['downstream']
    args.downstream = downstream_cfg['dataset']
    args.task_type = downstream_cfg['task_type'] if args.task_type is None else args.task_type
    downstream_kwargs = {
        'downstream': args.downstream,
        'task_type': args.task_type,
        'output_path': downstream_cfg['output_path'],
        'train_paths': [
            os.path.join(downstream_cfg['rep_dir'], downstream_cfg['train_path'].format(i=i))
            for i in range(downstream_cfg['n_models'])
        ],
        'test_paths': [
            os.path.join(downstream_cfg['rep_dir'], downstream_cfg['test_path'].format(i=i))
            for i in range(downstream_cfg['n_models'])
        ]
    }
    downstream_results_df = run_downstreams(**downstream_kwargs, load=True)

    # Evaluate the metrics, including NC.
    if args.skip_evaluation:
        return

    evaluator = ReliEvaluator()
    eval_cfg = config['evaluation']
    ens_cfg = eval_cfg['ensemble']
    for dist in eval_cfg['distances']:
        # paths for loading representations
        ref_rep_paths = [
            os.path.join(ens_cfg['dir'], ens_cfg['ref'].format(j=j))
            for j in range(ens_cfg['n_ens'])
        ]
        eval_rep_paths = [
            os.path.join(ens_cfg['dir'], ens_cfg['eval'].format(j=j))
            for j in range(ens_cfg['n_ens'])
        ]

        # eval
        logging.info(f'Start evaluation with {dist} distance')
        results = evaluate_score(
            methods=eval_cfg['methods'],
            ref_rep_paths=ref_rep_paths,
            eval_rep_paths=eval_rep_paths,
            dist=dist,
            seed=args.seed,
            use_numba=True,  # choose False if numba doesn't work
            **eval_cfg['params'],
        )
        results.update_configs({
            'dist': dist,
            'seed': args.seed,
            'timestamp': datetime.now().timestamp()
        })
        evaluator.merge(results)

    # Save / load the model uuid info
    output_dir = os.path.join(args.output_dir, ens_cfg['type'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{args.ssl}_{args.arch}_{args.pretrain}_{args.downstream}_{args.task_type}_seed{args.seed}"
    )

    if args.debug:
        df_model = evaluator.save_score_model(output_path)
    else:
        df_model = evaluator.load_model()

    # Evaluate the correlation
    model_id = -1  # downstream performance of ensemble
    downstream_errs = df_to_dict(downstream_results_df[downstream_results_df.model_idx == model_id].groupby('data_idx').mean()
                                 .drop(columns=['task_idx', 'model_idx']).sort_index())
    evaluator.eval(downstream_errs, model_id=model_id)
    evaluator.save_corr(df_model, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UQ-SSL script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
