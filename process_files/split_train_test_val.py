from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse


def move_to_partition(subjects_root_path, patients, partition):
    if not os.path.exists(os.path.join(subjects_root_path, partition)):
        os.mkdir(os.path.join(subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(subjects_root_path, patient)
        dest = os.path.join(subjects_root_path, partition, patient)
        shutil.move(src, dest)


def split_train_test_val(subjects_root_path="./data/root/"):
    # parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    # parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    # args, _ = parser.parse_known_args()

    test_set = set()
    with open(os.path.join(os.path.dirname(__file__), '../support_files/testset.csv'), "r") as test_set_file:
        for line in test_set_file:
            x, y = line.split(',')
            if int(y) == 1:
                test_set.add(x)

    val_set = set()
    with open(os.path.join(os.path.dirname(__file__), '../support_files/valset.csv'), "r") as val_set_file:
        for line in val_set_file:
            x, y = line.split(',')
            if int(y) == 1:
                val_set.add(x)

    folders = os.listdir(subjects_root_path)
    folders = list((filter(str.isdigit, folders)))
    train_patients = [x for x in folders if (x not in test_set) and (x not in val_set)]
    test_patients = [x for x in folders if x in test_set]
    val_patients = [x for x in folders if x in val_set]

    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(subjects_root_path, train_patients, "train")
    move_to_partition(subjects_root_path, test_patients, "test")
    move_to_partition(subjects_root_path, val_patients, "validation")
