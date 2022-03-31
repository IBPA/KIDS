import argparse
from glob import glob
import itertools
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
from utils import save_pkl  # noqa: E402


DEFAULT_ROOT_DATA_DIR = '../../../kg_constructor/output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_MODE = 'evaluate'
DEFAULT_BASED_ON = 'f1'
DEFAULT_RANDOM_STATE = 530
OUTPUT_DIR = '../../output'

FOLD = 'fold_0'


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET,
        help=f'Which KIDS dataset to use. (Default: {DEFAULT_DATASET}).',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default=DEFAULT_MODE,
        help=f'Either evaluate or final. (Default: {DEFAULT_MODE}).',
    )

    parser.add_argument(
        '--based_on',
        type=str,
        default=DEFAULT_BASED_ON,
        help=f'What metric to use for finding the best hyper-parameter. '
             f'(Default: {DEFAULT_BASED_ON})',
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f'Random seed. (Default: {DEFAULT_RANDOM_STATE})'
    )

    parser.add_argument(
        '--batch_size',
        type=str,
        required=True,
        help='Batch size.'
    )

    parser.add_argument(
        '--neg_ent',
        type=str,
        required=True,
        help='Negative entities.'
    )

    parser.add_argument(
        '--dim',
        type=str,
        required=True,
        help='Dimension.'
    )

    parser.add_argument(
        '--p_norm',
        type=str,
        required=True,
        help='P-norm..'
    )

    parser.add_argument(
        '--margin',
        type=str,
        required=True,
        help='Margin.'
    )

    parser.add_argument(
        '--adv_temperature',
        type=str,
        required=True,
        help='Adversarial temperature.'
    )

    parser.add_argument(
        '--alpha',
        type=str,
        required=True,
        help='Alpha.'
    )

    parser.add_argument(
        '--train_times',
        type=str,
        required=True,
        help='Iterations.'
    )

    parser.add_argument(
        '--n_workers',
        type=int,
        default=cpu_count() - 1,
        help='Number of workers for multiprocessing.'
    )

    parser.add_argument(
        '--chunksize',
        type=int,
        default=1000,
        help='Chunksize.'
    )

    parser.add_argument(
        '--output_filename',
        type=str,
        default=None,
        help='Output filename..'
    )

    args = parser.parse_args()

    return args


def load_data(filepath):
    with open(filepath, mode='r', encoding='utf-8') as _file:
        next(_file)  # skip header
        lines = _file.readlines()
    lines = [l.replace('\n', '') for l in lines]

    dataloader = []
    labels = []

    for line in lines:
        head, tail, rel, label = line.split(' ')
        dataloader.append(
            {'batch_h': np.array([int(head)]),
             'batch_t': np.array([int(tail)]),
             'batch_r': np.array([int(rel)]),
             'mode': 'normal'}
        )
        labels.append(1 if label == '1' else 0)

    return dataloader, np.array(labels)


def search_hyperparameter(args, hyperparameters, results_dir):
    # hyperparameter search mode
    print('Multiple sets of hyperparameters were passed.')
    print('Doing grid search using train/val set.')

    assert args.mode == 'evaluate'
    assert args.output_filename is not None

    data_dir = os.path.join(DEFAULT_ROOT_DATA_DIR, args.dataset, 'folds')
    all_folds = sorted(glob(f'{data_dir}/fold_*'))
    results = {}

    for idx, hp in enumerate(hyperparameters):
        print('*******************************')
        print(f'Processing {idx+1}/{len(hyperparameters)}...')
        print(f'Hyperparameters: {hp}')

        for fold in all_folds:
            print(f'Processing fold: {fold}')

            # dataloader for training
            train_dataloader = TrainDataLoader(
                in_path=None,
                tri_file=os.path.join(fold, 'train2id.txt'),
                ent_file=os.path.join(fold, 'entity2id.txt'),
                rel_file=os.path.join(fold, 'relation2id.txt'),
                batch_size=hp[0],
                threads=8,
                sampling_mode="normal",
                bern_flag=0,
                filter_flag=1,
                neg_ent=hp[1],
                neg_rel=0)

            # define the model
            transe = TransE(
                ent_tot=train_dataloader.get_ent_tot(),
                rel_tot=train_dataloader.get_rel_tot(),
                dim=hp[2],
                p_norm=hp[3],
                norm_flag=True,
                margin=hp[4])

            # define the loss function
            model = NegativeSampling(
                model=transe,
                loss=SigmoidLoss(adv_temperature=hp[5]),
                batch_size=train_dataloader.get_batch_size(),
                regul_rate=0.0
            )

            # train the model
            trainer = Trainer(
                model=model,
                data_loader=train_dataloader,
                train_times=hp[7],
                alpha=hp[6],
                use_gpu=True,
                opt_method='adam',
            )
            trainer.run()

            transe.save_checkpoint('./checkpoint/transe.ckpt')
            transe.load_checkpoint('./checkpoint/transe.ckpt')

            # get threshold using the validation set
            print('Running validation...')
            val_dataloader, val_labels = load_data(os.path.join(fold, 'val2id.txt'))
            tester = Tester(
                model=transe,
                data_loader=val_dataloader,
                use_gpu=True,
                n_workers=args.n_workers,
                chunksize=args.chunksize,
            )
            r = tester.evaluate_kids(labels=val_labels, mode=args.mode)
            if hp not in results:
                results[hp] = [r]
            else:
                results[hp].append(r)

    save_pkl(results, os.path.join(results_dir, args.output_filename))


def do_eval():
    all_files = glob('./checkpoint/transe/*.ckpt')

    results_dict = {
        'neg_ent': [],
        'dim': [],
        'margin': [],
        'adv_temperature': [],
        'alpha': [],
        'epoch': [],
        'f1': [],
    }

    for idx, file in enumerate(all_files):
        print(f'Evaluating {idx+1}/{len(all_files)}')
        print(f'Processing file: {file}')

        file_split = file.replace('.ckpt', '').split('/')[-1].split('-')
        neg_ent = file_split[0].replace('neg_ent', '')
        dim = file_split[1].replace('dim', '')
        margin = file_split[2].replace('margin', '')
        adv_temperature = file_split[3].replace('adv_temperature', '')
        alpha = file_split[4].replace('alpha', '')
        epoch = file_split[5]

        transe = TransE(
            ent_tot=8047,
            rel_tot=12,
            dim=int(dim),
            p_norm=2,
            norm_flag=True,
            margin=float(margin))

        transe.load_checkpoint(file)

        # get threshold using the validation set
        print('Running validation...')
        val_dataloader, val_labels = load_data(f'./benchmarks/KIDS/folds/{FOLD}/val2id.txt')
        tester = Tester(model=transe, data_loader=val_dataloader, use_gpu=True)
        f1, threshold, _ = tester.run_my_triple_classification(labels=val_labels, threshold=None)

        results_dict['neg_ent'].append(neg_ent)
        results_dict['dim'].append(dim)
        results_dict['margin'].append(margin)
        results_dict['adv_temperature'].append(adv_temperature)
        results_dict['alpha'].append(alpha)
        results_dict['epoch'].append(epoch)
        results_dict['f1'].append(f1)

    df_results = pd.DataFrame.from_dict(results_dict)
    df_results.to_csv('./results/transe_results.txt', index=False)


def do_5fold():

    for fold in range(5):
        print(f'Processing fold{fold}')

        # # dataloader for training
        # train_dataloader = TrainDataLoader(
        #     in_path=f'./benchmarks/KIDS/folds/fold_{fold}/',
        #     batch_size=512,
        #     threads=8,
        #     sampling_mode="normal",
        #     bern_flag=0,
        #     filter_flag=1,
        #     neg_ent=100,
        #     neg_rel=0)

        transe = TransE(
            ent_tot=8047,
            rel_tot=12,
            dim=256,
            p_norm=2,
            norm_flag=True,
            margin=12.0)

        # # define the loss function
        # model = NegativeSampling(
        #     model=transe,
        #     loss=SigmoidLoss(adv_temperature=1),
        #     batch_size=train_dataloader.get_batch_size(),
        #     regul_rate=0.0
        # )

        # # train the model
        # trainer = Trainer(
        #     model=model,
        #     data_loader=train_dataloader,
        #     train_times=200,
        #     alpha=0.001,
        #     use_gpu=True,
        #     opt_method='adam',
        #     save_steps=50,
        #     checkpoint_dir=f'./checkpoint/transe/fold{fold}'
        # )
        # trainer.run()

        # get threshold using the validation set
        print('Running validation...')
        val_dataloader, val_labels = load_data(f'./benchmarks/KIDS/folds/fold_{fold}/val2id.txt')
        transe.load_checkpoint(f'./checkpoint/transe/fold{fold}-49.ckpt')
        tester = Tester(model=transe, data_loader=val_dataloader, use_gpu=True)
        _, threshold, _ = tester.run_my_triple_classification(labels=val_labels, threshold=None)

        # run test using the threshold found above
        print('Running test...')
        test_dataloader, test_labels = load_data(f'./benchmarks/KIDS/folds/fold_{fold}/test2id.txt')
        transe.load_checkpoint(f'./checkpoint/transe/fold{fold}-49.ckpt')
        tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
        _, _, score = tester.run_my_triple_classification(labels=test_labels, threshold=threshold)

        df_score_and_label = pd.DataFrame(list(zip(test_labels, score)), columns=['label', 'score'])
        df_score_and_label.to_csv(f'./results/transe_fold{fold}.csv', sep='\t', index=False)


def main():
    args = parse_argument()

    # parse hyperparameters
    batch_size = [int(x) for x in args.batch_size.split(',')]
    neg_ent = [int(x) for x in args.neg_ent.split(',')]
    dim = [int(x) for x in args.dim.split(',')]
    p_norm = [int(x) for x in args.p_norm.split(',')]
    margin = [float(x) for x in args.margin.split(',')]
    adv_temperature = [int(x) for x in args.adv_temperature.split(',')]
    alpha = [float(x) for x in args.alpha.split(',')]
    train_times = [int(x) for x in args.train_times.split(',')]

    print(f'batch_size: {batch_size}')
    print(f'neg_ent: {neg_ent}')
    print(f'dim: {dim}')
    print(f'p_norm: {p_norm}')
    print(f'margin: {margin}')
    print(f'adv_temperature: {adv_temperature}')
    print(f'alpha: {alpha}')
    print(f'train_times: {train_times}')

    hyperparameters = list(itertools.product(
        batch_size,
        neg_ent,
        dim,
        p_norm,
        margin,
        adv_temperature,
        alpha,
        train_times,
    ))

    results_dir = os.path.join(OUTPUT_DIR, args.dataset, 'transe')
    if not Path(results_dir).exists():
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    if len(hyperparameters) > 1:
        search_hyperparameter(args, hyperparameters, results_dir)
    elif args.mode == 'evaluate':
        pass
        # data_dir = os.path.join(DEFAULT_ROOT_DATA_DIR, args.dataset, 'folds')
        # all_folds = sorted(glob(f'{data_dir}/fold_*'))
        # results = []

        # for fold in all_folds:
        #     data = Data(data_dir=fold, mode=args.mode, reverse=False)
        #     experiment = Experiment(
        #         data=data,
        #         n_workers=args.n_workers,
        #         chunksize=args.chunksize,
        #         num_iterations=num_iterations[0],
        #         batch_size=batch_size[0],
        #         learning_rate=learning_rate[0],
        #         decay_rate=decay_rate[0],
        #         ent_vec_dim=ent_vec_dim[0],
        #         rel_vec_dim=rel_vec_dim[0],
        #         cuda=args.cuda,
        #         input_dropout=input_dropout[0],
        #         hidden_dropout1=hidden_dropout1[0],
        #         hidden_dropout2=hidden_dropout2[0],
        #         label_smoothing=label_smoothing[0],
        #     )

        #     r = experiment.train_and_eval()
        #     results.append(r[1])

        # save_pkl(results, os.path.join(results_dir, 'evaluation_results.pkl'))
    elif args.mode == 'final':
        pass
        # data_dir = os.path.join(DEFAULT_ROOT_DATA_DIR, args.dataset, 'final')
        # data = Data(data_dir=data_dir, mode=args.mode, reverse=False)
        # experiment = Experiment(
        #     data=data,
        #     n_workers=args.n_workers,
        #     chunksize=args.chunksize,
        #     num_iterations=num_iterations[0],
        #     batch_size=batch_size[0],
        #     learning_rate=learning_rate[0],
        #     decay_rate=decay_rate[0],
        #     ent_vec_dim=ent_vec_dim[0],
        #     rel_vec_dim=rel_vec_dim[0],
        #     cuda=args.cuda,
        #     input_dropout=input_dropout[0],
        #     hidden_dropout1=hidden_dropout1[0],
        #     hidden_dropout2=hidden_dropout2[0],
        #     label_smoothing=label_smoothing[0],
        # )

        # score = experiment.train_and_eval(mode=args.mode)
        # df_results = pd.DataFrame(
        #     data.test_data,
        #     columns=['Subject', 'Predicate', 'Object', 'Label'])
        # df_results['score'] = score
        # df_results.sort_values('score', ascending=False, inplace=True)
        # df_results.to_csv(
        #     os.path.join(results_dir, 'hypothesis_score.txt'),
        #     sep='\t',
        #     index=False
        # )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')











    # if args.mode == 'search_hyperparameter':
    #     search_hyperparameter()
    # elif args.mode == 'evaluate':
    #     do_eval()
    # elif args.mode == '5fold':
    #     do_5fold()
    # else:
    #     raise ValueError('Invalid mode.')


if __name__ == '__main__':
    main()
