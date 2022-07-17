#!/usr/bin/python3
import numpy as np
import itertools

# ./test_parameter_generator.py > new_tests/test_parameters.txt
# split -l 32 new_tests/test_parameters.txt --numeric-suffixes --additional-suffix=.txt new_tests/test_parameters_
if __name__ == '__main__':
    template = "--experiment_dir_prefix {experiment_dir_prefix} --meta_train_n_way {meta_train_n_way} --meta_train_k_shot {meta_train_k_shot} --meta_train_query_count {meta_train_query_count} " \
               "--meta_train_max_epoch {meta_train_max_epoch} --meta_train_epoch_size {meta_train_epoch_size} --dataset_dir {dataset_dir} --model_x_dim0 {model_x_dim0} " \
               "--model_x_dim1 {model_x_dim1} --model_hid_dim {model_hid_dim} --model_z_dim {model_z_dim} --learning_rate {learning_rate} --learning_rate_decay {learning_rate_decay} " \
               "--meta_test_n_way {meta_test_n_way} --meta_test_k_shot {meta_test_k_shot} --meta_test_query_count {meta_test_query_count} --meta_test_episode_count {meta_test_episode_count} " \
               "--early_stop_change_acc_threshold {early_stop_change_acc_threshold} --early_stop_acc_window_length {early_stop_acc_window_length} " \
               "--early_stop_train_max_acc {early_stop_train_max_acc} "

    experiment_dir_prefix_range = ["prototypical_k_test_PS-B-FP-SP"]
    meta_train_n_way_range = [8]
    meta_train_k_shot_range = [5, 10, 15, 20, 25, 50]
    meta_train_query_count_range = [64, 128, 256, 420]
    meta_train_max_epoch_range = [256]
    meta_train_epoch_size_range = [256]
    dataset_dir_range = ["../../datasets/CIC_IDS_2017/cic_ids_2017_8t470_4t470_12v130_PS-B-FP-SP"]
    model_x_dim0_range = [1]
    model_x_dim1_range = [78]
    model_hid_dim_range = [64, 128]
    model_z_dim_range = [64, 128]
    learning_rate_range = [0.001]
    learning_rate_decay_range = [0.1]
    meta_test_n_way_range = [4]
    meta_test_k_shot_range = [5, 10, 15, 20, 25, 50]
    meta_test_query_count_range = [470]
    meta_test_episode_count_range = [1000]
    early_stop_change_acc_threshold_range = [0.00022]
    early_stop_acc_window_length_range = [5]
    early_stop_train_max_acc_range = [0.98]

    for experiment_dir_prefix, meta_train_n_way, meta_train_k_shot, meta_train_query_count, meta_train_max_epoch, meta_train_epoch_size, dataset_dir, model_x_dim0, model_x_dim1, model_hid_dim, \
        model_z_dim, learning_rate, learning_rate_decay, meta_test_n_way, meta_test_k_shot, \
        meta_test_query_count, meta_test_episode_count, early_stop_change_acc_threshold, early_stop_acc_window_length, early_stop_train_max_acc in itertools.product(experiment_dir_prefix_range,
                                                                                                                                                                     meta_train_n_way_range,
                                                                                                                                                                     meta_train_k_shot_range,
                                                                                                                                                                     meta_train_query_count_range,
                                                                                                                                                                     meta_train_max_epoch_range,
                                                                                                                                                                     meta_train_epoch_size_range,
                                                                                                                                                                     dataset_dir_range,
                                                                                                                                                                     model_x_dim0_range,
                                                                                                                                                                     model_x_dim1_range,
                                                                                                                                                                     model_hid_dim_range,
                                                                                                                                                                     model_z_dim_range,
                                                                                                                                                                     learning_rate_range,
                                                                                                                                                                     learning_rate_decay_range,
                                                                                                                                                                     meta_test_n_way_range,
                                                                                                                                                                     meta_test_k_shot_range,
                                                                                                                                                                     meta_test_query_count_range,
                                                                                                                                                                     meta_test_episode_count_range,
                                                                                                                                                                     early_stop_change_acc_threshold_range,
                                                                                                                                                                     early_stop_acc_window_length_range,
                                                                                                                                                                     early_stop_train_max_acc_range
                                                                                                                                                                     ):
        generated = template.format(
            experiment_dir_prefix=experiment_dir_prefix,
            meta_train_n_way=meta_train_n_way,
            meta_train_k_shot=meta_train_k_shot,
            meta_train_query_count=meta_train_query_count,
            meta_train_max_epoch=meta_train_max_epoch,
            meta_train_epoch_size=meta_train_epoch_size,
            dataset_dir=dataset_dir,
            model_x_dim0=model_x_dim0,
            model_x_dim1=model_x_dim1,
            model_hid_dim=model_hid_dim,
            model_z_dim=model_z_dim,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            meta_test_n_way=meta_test_n_way,
            meta_test_k_shot=meta_test_k_shot,
            meta_test_query_count=meta_test_query_count - meta_test_k_shot,
            meta_test_episode_count=meta_test_episode_count,
            early_stop_change_acc_threshold=early_stop_change_acc_threshold,
            early_stop_acc_window_length=early_stop_acc_window_length,
            early_stop_train_max_acc=early_stop_train_max_acc
        )
        print(generated)
