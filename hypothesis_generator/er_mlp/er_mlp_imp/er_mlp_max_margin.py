"""
Filename: er_mlp_max_margin.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Construct ER MLP using max margin loss and perform train, evaluation, and test.

To-do:
"""
# standard imports
import logging as log
import os
import pickle
import random

# third party imports
import numpy as np
import tensorflow as tf

# local imports
from data_processor import DataProcessor
from er_mlp import ERMLP
from metrics import plot_cost, plot_map


def run_model(params, final_model=False):
    """
    Run the ER_MLP model using max margin loss.

    Inputs:
        params: dictionary containing different
            parameters to be used when running the model
    """
    ######################
    # data preprocessing #
    ######################
    processor = DataProcessor()

    # load data
    train_df = processor.load(os.path.join(params['data_path'], params['train_file']))
    train_local_df = processor.load(os.path.join(params['data_path'], 'train_local.txt'))

    log.debug('train dataframe shape: %s', train_df.shape)
    log.debug('train_local dataframe shape: %s', train_local_df.shape)

    if not final_model:
        dev_df = processor.load(os.path.join(params['data_path'], 'dev.txt'))
        test_df = processor.load(os.path.join(params['data_path'], 'test.txt'))

        log.debug('dev dataframe shape: %s', dev_df.shape)
        log.debug('test dataframe shape: %s', test_df.shape)

    # make sure we have label column
    if len(train_df.columns) < 4:
        log.warning('Label (last column) is missing')
        train_df['one'] = 1

    # do word embeddings
    if params['word_embedding']:
        indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(
            os.path.join(params['data_path'], 'entities.txt'))
        indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(
            os.path.join(params['data_path'], 'relations.txt'))
    else:
        entity_dic = processor.machine_translate(os.path.join(params['data_path'], 'entities.txt'))
        pred_dic = processor.machine_translate(os.path.join(params['data_path'], 'relations.txt'))

    # numerically represent the data
    indexed_train_data = processor.create_indexed_triplets_with_label(
        train_df.values, entity_dic, pred_dic)
    indexed_train_local_data = processor.create_indexed_triplets_with_label(
        train_local_df.values, entity_dic, pred_dic)

    if not final_model:
        indexed_dev_data = processor.create_indexed_triplets_with_label(
            dev_df.values, entity_dic, pred_dic)
        indexed_test_data = processor.create_indexed_triplets_with_label(
            test_df.values, entity_dic, pred_dic)

    # change label from -1 to 0
    if not final_model:
        indexed_train_local_data[:, 3][indexed_train_local_data[:, 3] == -1] = 0
        indexed_dev_data[:, 3][indexed_dev_data[:, 3] == -1] = 0
        indexed_test_data[:, 3][indexed_test_data[:, 3] == -1] = 0

        # shuffle test data
        np.random.shuffle(indexed_train_local_data)
        np.random.shuffle(indexed_test_data)

    # construct new parameter dictionary to be fed into the network
    er_mlp_params = {
        'word_embedding': params['word_embedding'],
        'embedding_size': params['embedding_size'],
        'layer_size': params['layer_size'],
        'corrupt_size': params['corrupt_size'],
        'lambda': params['lambda'],
        'num_entities': len(entity_dic),
        'num_preds': len(pred_dic),
        'learning_rate': params['learning_rate'],
        'batch_size': params['batch_size'],
        'add_layers': params['add_layers'],
        'act_function': params['act_function'],
        'drop_out_percent': params['drop_out_percent'],
        'margin': params['margin']
    }

    # append word embedding related parameters to the dictionary
    if params['word_embedding']:
        er_mlp_params['num_entity_words'] = num_entity_words
        er_mlp_params['num_pred_words'] = num_pred_words
        er_mlp_params['indexed_entities'] = indexed_entities
        er_mlp_params['indexed_predicates'] = indexed_predicates

    #########################
    # construct the network #
    #########################
    er_mlp = ERMLP(er_mlp_params)

    # network used for training
    train_predictions = er_mlp.build_traininig_model()
    tf.add_to_collection('train_predictions', train_predictions)

    # network used for testing
    test_predictions = er_mlp.build_testing_model()
    tf.add_to_collection('test_predictions', test_predictions)

    # loss
    cost = er_mlp.loss()
    tf.add_to_collection('cost', cost)
    tf.summary.scalar('cost', cost)

    # optimizer
    if params['optimizer'] == 0:
        optimizer = er_mlp.train_adagrad(cost)  # adagrad
    else:
        optimizer = er_mlp.train_adam(cost)  # adam

    tf.add_to_collection('optimizer', optimizer)

    # merge summary
    merged = tf.summary.merge_all()

    # saver to save the model
    saver = tf.train.Saver()

    # choose the positive training data
    data_train = indexed_train_data[indexed_train_data[:, 3] == 1]
    data_train = data_train[:, :3]

    # some variable initializations
    iter_list = []
    cost_list = []
    train_local_map_list = []
    test_map_list = []
    iteration = 0

    # init variables
    log.info('Initializing tensor variables...')
    init_all = tf.global_variables_initializer()

    #########################
    # train the network #
    #########################
    log.info('Begin training...')

    # begin session
    with tf.Session() as sess:
        # writer
        train_writer = tf.summary.FileWriter(
            os.path.join(params['model_save_directory'], 'log'),
            sess.graph)

        # run init
        sess.run(init_all)

        # epoch
        for epoch in range(params['training_epochs']):
            log.info('****** Epoch: %d/%d ******', epoch, params['training_epochs'])

            total_batch = int(np.ceil(data_train.shape[0] / params['batch_size']))

            # shuffle the training data for each epoch
            np.random.shuffle(data_train)

            # iteration
            for i in range(total_batch):
                # get corrupted batch using the un-corrupted data_train
                start_idx = i * params['batch_size']
                end_idx = (i + 1) * params['batch_size']
                batch_xs = er_mlp.get_training_batch_with_corrupted(data_train[start_idx:end_idx])

                # flip bit
                flip = bool(random.getrandbits(1))

                # feed dictionary
                feed_dict = {
                    er_mlp.train_triplets: batch_xs,
                    er_mlp.flip_placeholder: flip}

                # display progress
                if (i == 0) and (epoch % params['display_step'] == 0):
                    _, train_summary, current_cost = sess.run(
                        [optimizer, merged, cost],
                        feed_dict=feed_dict)

                    train_writer.add_summary(train_summary, iteration)

                    log.info('current cost: %f', current_cost)

                    train_local_map = er_mlp.test_model(
                        sess,
                        indexed_train_local_data,
                        pred_dic,
                        _type='train local')

                    train_local_map_list.append(train_local_map)

                    if not final_model:
                        thresholds = er_mlp.determine_threshold(
                            sess,
                            indexed_dev_data,
                            use_f1=params['f1_for_threshold'])

                        test_map = er_mlp.test_model(
                            sess,
                            indexed_test_data,
                            pred_dic,
                            threshold=thresholds,
                            _type='current test')

                        test_map_list.append(test_map)

                    iter_list.append(iteration)
                    cost_list.append(current_cost)
                else:
                    sess.run(optimizer, feed_dict=feed_dict)

                # update iteration
                iteration += 1

        # close writers
        train_writer.close()

        # do final threshold determination and testing model
        if not final_model:
            log.info('determine threshold for classification')

            thresholds = er_mlp.determine_threshold(
                sess,
                indexed_dev_data,
                use_f1=params['f1_for_threshold'])

            er_mlp.test_model(
                sess,
                indexed_test_data,
                pred_dic,
                threshold=thresholds,
                _type='final')

        # plot the cost graph
        plot_cost(
            iter_list,
            cost_list,
            params['model_save_directory'])

        plot_map(
            iter_list,
            train_local_map_list,
            params['model_save_directory'],
            filename='train_local_map.png')

        if not final_model:
            plot_map(
                iter_list,
                test_map_list,
                params['model_save_directory'],
                filename='map.png')

        # save the model & parameters if prompted
        if params['save_model']:
            saver.save(sess, os.path.join(params['model_save_directory'], 'model'))
            log.info('model saved in: %s', params['model_save_directory'])

            save_object = {
                'entity_dic': entity_dic,
                'pred_dic': pred_dic
            }

            if not final_model:
                save_object['thresholds'] = thresholds

            if params['word_embedding']:
                save_object['indexed_entities'] = indexed_entities
                save_object['indexed_predicates'] = indexed_predicates
                save_object['num_pred_words'] = num_pred_words
                save_object['num_entity_words'] = num_entity_words

            with open(os.path.join(params['model_save_directory'], 'params.pkl'), 'wb') as output:
                pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
