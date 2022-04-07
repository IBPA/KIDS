import argparse
from collections import defaultdict
from functools import partial
from glob import glob
import itertools
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path
import sys
sys.path.append("..")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from torch.optim.lr_scheduler import ExponentialLR  # noqa: E402
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score  # noqa: E402

from load_data import Data  # noqa: E402
from model import TuckER  # noqa: E402
from utils import save_pkl, load_pkl  # noqa: E402

DEFAULT_ROOT_DATA_DIR = '../../../kg_constructor/output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_MODE = 'evaluate'
DEFAULT_BASED_ON = 'f1'
DEFAULT_RANDOM_STATE = 530
OUTPUT_DIR = '../../output'


class Experiment:

    def __init__(self, data, chunksize, n_workers, learning_rate=0.0005,
                 ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.data = data
        self.n_workers = n_workers
        self.chunksize = chunksize
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(self.data.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def _calculate_f1(self, predictions, gt_labels, idx):
        local_predictions = (predictions >= predictions[idx]) * 1
        return f1_score(gt_labels, local_predictions)

    def get_threshold(self, score, labels):
        both = np.column_stack((score, labels))
        both = both[both[:, 0].argsort()]

        # get a flattened array after the sort
        predictions = both[:, 0].ravel()
        gt_labels = both[:, 1].ravel()

        # f1_scores = np.zeros(np.shape(predictions))
        f1_scores = []
        idx_list = list(range(np.shape(predictions)[0]))
        with Pool(args.n_workers) as p:
            for f1 in list(tqdm(p.imap(
                        partial(self._calculate_f1, predictions, gt_labels),
                        idx_list,
                        chunksize=self.chunksize),
                    total=len(idx_list))):
                f1_scores.append(f1)

        # find all the indices that has the best f1
        indices = np.argmax(f1_scores)
        threshold = np.mean(predictions[indices])

        return threshold

    def evaluate(self, model, data, mode):
        print(f'Mode: {mode}')
        test_data_idxs_all = np.array(self.get_data_idxs(data))

        score = []
        for idx in range(0, test_data_idxs_all.shape[0], 10000):
            if idx + 10000 > test_data_idxs_all.shape[0]:
                test_data_idxs = test_data_idxs_all[idx:, ]
            else:
                test_data_idxs = test_data_idxs_all[idx:idx+10000,]
        
            e1_idx = torch.tensor(test_data_idxs[:, 0])
            r_idx = torch.tensor(test_data_idxs[:, 1])
            e2_idx = torch.tensor(test_data_idxs[:, 2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            try:
                y_pred = model.forward(e1_idx, r_idx)
            except RuntimeError as err:
                print(f'Runtime error: {err}')

                assert e1_idx.size()[0] == r_idx.size()[0]
                y_pred = []
                size = e1_idx.size()[0]
                for i in tqdm(range(size)):
                    y_pred.append(model.forward(e1_idx[[i]], r_idx[[i]]))
                y_pred = torch.cat(y_pred)

            for idx, x in enumerate(test_data_idxs[:, 2]):
                score.append(y_pred[idx, x].item())

        if mode == 'final':
            return score

        labels = [1 if x[-1] == '1' else 0 for x in data]
        threshold = self.get_threshold(score, labels)
        predictions = (score >= threshold) * 1

        f1 = f1_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        print(type(labels))
        print(type(predictions))
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)

        metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision,
            'recall': recall,
        }

        print(f'f1: {f1}')
        print(f'accuracy: {accuracy}')
        print(f'confusion: {tp} {fp} {fn} {tn}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')

        return score, metrics, threshold

    def train_and_eval(self, mode=None):
        print("Training the TuckER model...")
        self.entity_idxs = {self.data.entities[i]: i for i in range(len(self.data.entities))}
        self.relation_idxs = {self.data.relations[i]: i for i in range(len(self.data.relations))}

        train_data_idxs = self.get_data_idxs(self.data.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(self.data, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(f'Iteration: {it}')
            print(f'Loss: {np.mean(losses)}')

        model.eval()
        with torch.no_grad():
            if mode != 'final':
                score, metrics, threshold = self.evaluate(model, self.data.test_data, mode=mode)
                return score, metrics, threshold
            elif mode == 'final':
                score = self.evaluate(model, self.data.test_data, mode=mode)
                return score
            else:
                raise ValueError(f'Invalid mode: {mode}')


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
        '--num_iterations',
        type=str,
        # default=300,
        help='Number of iterations.'
    )

    parser.add_argument(
        '--batch_size',
        type=str,
        # default=128,
        help='Batch size.'
    )

    parser.add_argument(
        '--lr',
        type=str,
        # default=0.0002,
        help='Learning rate.'
    )

    parser.add_argument(
        '--dr',
        type=str,
        # default=1.0,
        help='Decay rate.'
    )

    parser.add_argument(
        '--edim',
        type=str,
        # default=200,
        help='Entity embedding dimensionality.'
    )

    parser.add_argument(
        '--rdim',
        type=str,
        # default=30,
        help='Relation embedding dimensionality.'
    )

    parser.add_argument(
        '--cuda',
        type=bool,
        default=True,
        help='Whether to use cuda (GPU) or not (CPU).'
    )

    parser.add_argument(
        '--input_dropout',
        type=str,
        # default=0.2,
        help='Input layer dropout.'
    )

    parser.add_argument(
        '--hidden_dropout1',
        type=str,
        # default=0.4,
        help='Dropout after the first hidden layer.'
    )

    parser.add_argument(
        '--hidden_dropout2',
        type=str,
        # default=0.5,
        help='Dropout after the second hidden layer.'
    )

    parser.add_argument(
        '--label_smoothing',
        type=str,
        # default=0.1,
        help='Amount of label smoothing.'
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


if __name__ == '__main__':
    args = parse_argument()

    # parse hyperparameters
    num_iterations = [int(x) for x in args.num_iterations.split(',')]
    batch_size = [int(x) for x in args.batch_size.split(',')]
    learning_rate = [float(x) for x in args.lr.split(',')]
    decay_rate = [float(x) for x in args.dr.split(',')]
    ent_vec_dim = [int(x) for x in args.edim.split(',')]
    rel_vec_dim = [int(x) for x in args.rdim.split(',')]
    input_dropout = [float(x) for x in args.input_dropout.split(',')]
    hidden_dropout1 = [float(x) for x in args.hidden_dropout1.split(',')]
    hidden_dropout2 = [float(x) for x in args.hidden_dropout2.split(',')]
    label_smoothing = [float(x) for x in args.label_smoothing.split(',')]

    print(f'num_iterations: {num_iterations}')
    print(f'batch_size: {batch_size}')
    print(f'learning_rate: {learning_rate}')
    print(f'decay_rate: {decay_rate}')
    print(f'ent_vec_dim: {ent_vec_dim}')
    print(f'rel_vec_dim: {rel_vec_dim}')
    print(f'input_dropout: {input_dropout}')
    print(f'hidden_dropout1: {hidden_dropout1}')
    print(f'hidden_dropout2: {hidden_dropout2}')
    print(f'label_smoothing: {label_smoothing}')

    hyperparameters = list(itertools.product(
        num_iterations,
        batch_size,
        learning_rate,
        decay_rate,
        ent_vec_dim,
        rel_vec_dim,
        input_dropout,
        hidden_dropout1,
        hidden_dropout2,
        label_smoothing,
    ))

    torch.backends.cudnn.deterministic = True
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.random_state)

    results_dir = os.path.join(OUTPUT_DIR, args.dataset, 'tucker')
    if not Path(results_dir).exists():
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    if len(hyperparameters) > 1:
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

                # train / validation
                print('Doing training and validation')
                data = Data(data_dir=fold, mode='gridsearch', reverse=False)
                experiment = Experiment(
                    data=data,
                    n_workers=args.n_workers,
                    chunksize=args.chunksize,
                    num_iterations=hp[0],
                    batch_size=hp[1],
                    learning_rate=hp[2],
                    decay_rate=hp[3],
                    ent_vec_dim=hp[4],
                    rel_vec_dim=hp[5],
                    cuda=args.cuda,
                    input_dropout=hp[6],
                    hidden_dropout1=hp[7],
                    hidden_dropout2=hp[8],
                    label_smoothing=hp[9],
                )

                r = experiment.train_and_eval()
                if hp not in results:
                    results[hp] = [r]
                else:
                    results[hp].append(r)

        save_pkl(results, os.path.join(results_dir, args.output_filename))
    elif args.mode == 'evaluate':
        data_dir = os.path.join(DEFAULT_ROOT_DATA_DIR, args.dataset, 'folds')
        all_folds = sorted(glob(f'{data_dir}/fold_*'))
        results = []

        for fold in all_folds:
            data = Data(data_dir=fold, mode=args.mode, reverse=False)
            experiment = Experiment(
                data=data,
                n_workers=args.n_workers,
                chunksize=args.chunksize,
                num_iterations=num_iterations[0],
                batch_size=batch_size[0],
                learning_rate=learning_rate[0],
                decay_rate=decay_rate[0],
                ent_vec_dim=ent_vec_dim[0],
                rel_vec_dim=rel_vec_dim[0],
                cuda=args.cuda,
                input_dropout=input_dropout[0],
                hidden_dropout1=hidden_dropout1[0],
                hidden_dropout2=hidden_dropout2[0],
                label_smoothing=label_smoothing[0],
            )

            r = experiment.train_and_eval()
            results.append(r[1])

        save_pkl(results, os.path.join(results_dir, 'evaluation_results.pkl'))
    elif args.mode == 'final':
        data_dir = os.path.join(DEFAULT_ROOT_DATA_DIR, args.dataset, 'final')
        data = Data(data_dir=data_dir, mode=args.mode, reverse=False)
        experiment = Experiment(
            data=data,
            n_workers=args.n_workers,
            chunksize=args.chunksize,
            num_iterations=num_iterations[0],
            batch_size=batch_size[0],
            learning_rate=learning_rate[0],
            decay_rate=decay_rate[0],
            ent_vec_dim=ent_vec_dim[0],
            rel_vec_dim=rel_vec_dim[0],
            cuda=args.cuda,
            input_dropout=input_dropout[0],
            hidden_dropout1=hidden_dropout1[0],
            hidden_dropout2=hidden_dropout2[0],
            label_smoothing=label_smoothing[0],
        )

        score = experiment.train_and_eval(mode=args.mode)
        df_results = pd.DataFrame(
            data.test_data,
            columns=['Subject', 'Predicate', 'Object', 'Label'])
        df_results['score'] = score
        df_results.sort_values('score', ascending=False, inplace=True)
        df_results.to_csv(
            os.path.join(results_dir, 'hypothesis_score.txt'),
            sep='\t',
            index=False
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
