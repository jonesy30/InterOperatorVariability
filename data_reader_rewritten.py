from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import sys
import numpy as np
import platform
import pickle
import json

class Discretizer:
    def __init__(self, timestep=0.8, store_masks=True, start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), './support_files/discretizer_config.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        prev_values = [[] for i in range(len(self._id_to_channel))]
        for bin_id in range(N_bins):
            for channel in self._id_to_channel:
                channel_id = self._channel_to_id[channel]
                if mask[bin_id][channel_id] == 1:
                    prev_values[channel_id].append(original_value[bin_id][channel_id])
                    continue

                if len(prev_values[channel_id]) == 0:
                    imputed_value = self._normal_values[channel]
                else:
                    imputed_value = prev_values[channel_id][-1]
                write(data, bin_id, channel, imputed_value, begin_pos)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret

class InHospitalMortalityReader(object):
    
    def get_number_of_examples(self):
        return len(self._data)

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)

    def __init__(self, dataset_dir, listfile):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """

        self._dataset_dir = dataset_dir
        self._current_index = 0

        listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]
        self._data = [[x,y] for (x, y, *rest) in self._data]

        self._data = [(x, int(y.strip())) for (x, y) in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return np.stack(ret)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        y = self._data[index][1]
        X = self._read_timeseries(name)

        return {"X": X,
                "y": y}

def load_data(data_directory, mode='train'):
    if mode == 'train':
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_directory, 'train'), listfile=os.path.join(data_directory, mode, 'listfile.csv'))
    elif mode == 'test':
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_directory, 'test'), listfile=os.path.join(data_directory, mode, 'listfile.csv'))
    elif mode == 'validation':
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_directory, 'validation'), listfile=os.path.join(data_directory, mode, 'listfile.csv'))
    else:
        raise ValueError("Invalid mode for load_data, set mode to either 'train', 'test' or 'validation'")

    discretizer = Discretizer(timestep=float(1), store_masks=True, start_time='zero')

    normalizer_state = './re_written_scripts/support_files/normalizer_file.normalizer'

    discretizer_header = discretizer.transform(reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer.load_params(normalizer_state)

    N = reader.get_number_of_examples()
    data_ret = {}
    for i in range(N):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data_ret:
                data_ret[k] = []
            data_ret[k].append(v)

    data = data_ret["X"]
    labels = data_ret["y"]

    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, [48]*len(data))]
    data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    return whole_data