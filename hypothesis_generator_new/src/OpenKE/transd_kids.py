import argparse
from glob import glob
import itertools
import numpy as np
import pandas as pd

from openke.config import Trainer, Tester
from openke.module.model import TransD
from openke.module.loss import SigmoidLoss, MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader


FOLD = 'fold_0'


def parse_argument():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Train/validate/test TransD on KIDS.')

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='search_hyperparameter | evaluate |  5fold.')

    return parser.parse_args()


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


def search_hyperparameter():
    neg_ent = [25, 50, 100]
    dim = [128, 256, 512, 1024]
    margin = [6.0, 12.0, 24.0]
    adv_temperature = [1]
    alpha = [0.001, 0.0001]

    hyperparameters = list(itertools.product(
        neg_ent,
        dim,
        margin,
        adv_temperature,
        alpha
    ))

    all_files = glob('./checkpoint/transd/*.ckpt')
    all_files = [x.split('/')[-1] for x in all_files]

    for idx, hyperparameter in enumerate(hyperparameters):
        print(f'Processing {idx+1}/{len(hyperparameters)}')

        neg_ent = hyperparameter[0]
        dim = hyperparameter[1]
        margin = hyperparameter[2]
        adv_temperature = hyperparameter[3]
        alpha = hyperparameter[4]

        filename = f'neg_ent{neg_ent}-dim{dim}-margin{margin}-' + \
                   f'adv_temperature{adv_temperature}-alpha{alpha}'

        print(f'Filename: {filename}')

        # dataloader for training
        train_dataloader = TrainDataLoader(
            in_path=f'./benchmarks/KIDS/folds/{FOLD}/',
            nbatches=32,
            threads=8,
            sampling_mode="normal",
            bern_flag=0,
            filter_flag=1,
            neg_ent=neg_ent,
            neg_rel=0)

        # define the model
        transd = TransD(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim_e=dim,
            dim_r=dim,
            p_norm=2,
            norm_flag=True,
            margin=margin)

        # define the loss function
        model = NegativeSampling(
            model=transd,
            # loss=MarginLoss(margin=4.0),
            loss=SigmoidLoss(adv_temperature=adv_temperature),
            batch_size=train_dataloader.get_batch_size(),
        )

        # train the model
        trainer = Trainer(
            model=model,
            data_loader=train_dataloader,
            train_times=200,
            alpha=alpha,
            use_gpu=True,
            opt_method='adam',
            save_steps=25,
            checkpoint_dir=f'./checkpoint/transd/{filename}'
        )
        trainer.run()


def do_eval():
    all_files = glob('./checkpoint/transd/*.ckpt')

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

        transd = TransD(
            ent_tot=8047,
            rel_tot=12,
            dim_e=int(dim),
            dim_r=int(dim),
            p_norm=2,
            norm_flag=True,
            margin=float(margin))

        transd.load_checkpoint(file)

        # get threshold using the validation set
        print('Running validation...')
        val_dataloader, val_labels = load_data(f'./benchmarks/KIDS/folds/{FOLD}/val2id.txt')
        tester = Tester(model=transd, data_loader=val_dataloader, use_gpu=True)
        f1, threshold, _ = tester.run_my_triple_classification(labels=val_labels, threshold=None)

        results_dict['neg_ent'].append(neg_ent)
        results_dict['dim'].append(dim)
        results_dict['margin'].append(margin)
        results_dict['adv_temperature'].append(adv_temperature)
        results_dict['alpha'].append(alpha)
        results_dict['epoch'].append(epoch)
        results_dict['f1'].append(f1)

    df_results = pd.DataFrame.from_dict(results_dict)
    df_results.to_csv('./results/transd_results.txt', index=False)


def do_5fold():

    for fold in range(5):
        print(f'Processing fold{fold}')

        # # dataloader for training
        # train_dataloader = TrainDataLoader(
        #     in_path=f'./benchmarks/KIDS/folds/fold_{fold}/',
        #     nbatches=8,
        #     threads=8,
        #     sampling_mode="normal",
        #     bern_flag=0,
        #     filter_flag=1,
        #     neg_ent=100,
        #     neg_rel=0)

        transd = TransD(
            ent_tot=8047,
            rel_tot=12,
            dim_e=256,
            dim_r=256,
            p_norm=2,
            norm_flag=True,
            margin=24.0)

        # # define the loss function
        # model = NegativeSampling(
        #     model=transd,
        #     loss=SigmoidLoss(adv_temperature=1),
        #     batch_size=train_dataloader.get_batch_size(),
        # )

        # # train the model
        # trainer = Trainer(
        #     model=model,
        #     data_loader=train_dataloader,
        #     train_times=200,
        #     alpha=0.0001,
        #     use_gpu=True,
        #     opt_method='adam',
        #     save_steps=100,
        #     checkpoint_dir=f'./checkpoint/transd/fold{fold}'
        # )
        # trainer.run()

        # get threshold using the validation set
        print('Running validation...')
        val_dataloader, val_labels = load_data(f'./benchmarks/KIDS/folds/fold_{fold}/val2id.txt')
        transd.load_checkpoint(f'./checkpoint/transd/fold{fold}-99.ckpt')
        tester = Tester(model=transd, data_loader=val_dataloader, use_gpu=True)
        _, threshold, _ = tester.run_my_triple_classification(labels=val_labels, threshold=None)

        # run test using the threshold found above
        print('Running test...')
        test_dataloader, test_labels = load_data(f'./benchmarks/KIDS/folds/fold_{fold}/test2id.txt')
        transd.load_checkpoint(f'./checkpoint/transd/fold{fold}-99.ckpt')
        tester = Tester(model=transd, data_loader=test_dataloader, use_gpu=True)
        _, _, score = tester.run_my_triple_classification(labels=test_labels, threshold=threshold)

        df_score_and_label = pd.DataFrame(list(zip(test_labels, score)), columns=['label', 'score'])
        df_score_and_label.to_csv(f'./results/transd_fold{fold}.csv', sep='\t', index=False)


def main():
    args = parse_argument()

    if args.mode == 'search_hyperparameter':
        search_hyperparameter()
    elif args.mode == 'evaluate':
        do_eval()
    elif args.mode == '5fold':
        do_5fold()
    else:
        raise ValueError('Invalid mode.')


if __name__ == '__main__':
    main()
