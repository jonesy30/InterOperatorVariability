Repository contains code for converting MIMIC-III data into data which can be read in by a machine learning model.
Please note, this code has been written for a Windows machine. Linux or Mac users may need to change directions of \'s in filepaths

This code is based off the work by Harutyunyan, et al., Multitask learning and benchmarking with clinical time series data, https://doi.org/10.1038/s41597-019-0103-9

For the original code on GitHub, please refer to their repository: https://github.com/YerevaNN/mimic3-benchmarks

TO RUN
run (from command line)
    python process_mimic.py [path_to_mimic_data]
where [path_to_mimic_data] is a filepath to a folder containing the extracted CSV files of the MIMIC-III dataset
NOTE: this process could take up to 7 hours to run

This code will create a folder in the same directory as this README, which will contain 4 folders:
    - root
        - ignore this - these are files used by the process_mimic script. 
        - You can delete if you want to, but if your data gets corrupted, it will save you a lot of time having these already!
    - train
        - Training data
    - test
        - Testing data
    - validation
        - Validation data

TO USE DATA
In Python script:
from data_reader_rewritten import load_data

train_raw = load_data(data_dir, mode='train')

train_data = np.asarray(train_raw[0])
train_labels = np.asarray(train_raw[1])

An example of this can be found in the example_network.py script

OTHER FILES
process_files
    - Scripts which are called from process_mimic.py
support_files
    - Files which are used by process_mimic (normalizer files, test set definitions, etc)