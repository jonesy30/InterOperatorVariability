from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml

# from mimic3benchmark.mimic3csv import *
from .preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from .util import dataframe_from_csv
from .mimic3csv import *
import os

def extract_subjects(mimic3_path, output_path="./data/root/", verbose=True):
    # parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
    # parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
    # parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
    # parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
    #                     default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
    # parser.add_argument('--phenotype_definitions', '-p', type=str,
    #                     default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
    #                     help='YAML file with phenotype definitions.')
    # parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
    # parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
    # parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
    # parser.set_defaults(verbose=True)
    # parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
    # args, _ = parser.parse_known_args()

    # try:
    #     os.makedirs(args.output_path)
    # except:
    #     pass

    itemids_file = None
    event_tables = ['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS']

    phenotype_definitions = os.path.join(os.path.dirname(__file__), '../support_files/hcup_ccs_2015_definitions.yaml')

    patients = read_patients_table(mimic3_path)
    admits = read_admissions_table(mimic3_path)
    stays = read_icustays_table(mimic3_path)
    if verbose:
        print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

    stays = remove_icustays_with_transfers(stays)
    if verbose:
        print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

    stays = merge_on_subject_admission(stays, admits)
    stays = merge_on_subject(stays, patients)
    stays = filter_admissions_on_nb_icustays(stays)
    if verbose:
        print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

    stays = add_age_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = filter_icustays_on_age(stays)
    if verbose:
        print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
            stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

    stays.to_csv(os.path.join(output_path, 'all_stays.csv'), index=False)
    diagnoses = read_icd_diagnoses_table(mimic3_path)
    diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
    diagnoses.to_csv(os.path.join(output_path, 'all_diagnoses.csv'), index=False)
    count_icd_codes(diagnoses, output_path=os.path.join(output_path, 'diagnosis_counts.csv'))

    phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(phenotype_definitions, 'r')))
    make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(output_path, 'phenotype_labels.csv'),
                                                        index=False, quoting=csv.QUOTE_NONNUMERIC)

    # if test:
    #     pat_idx = np.random.choice(patients.shape[0], size=1000)
    #     patients = patients.iloc[pat_idx]
    #     stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    #     event_tables = [event_tables[0]]
    #     print('Using only', stays.shape[0], 'stays and only', event_tables[0], 'table')

    subjects = stays.SUBJECT_ID.unique()
    break_up_stays_by_subject(stays, output_path, subjects=subjects)
    break_up_diagnoses_by_subject(phenotypes, output_path, subjects=subjects)
    items_to_keep = set(
        [int(itemid) for itemid in dataframe_from_csv(itemids_file)['ITEMID'].unique()]) if itemids_file else None
    for table in event_tables:
        read_events_table_and_break_up_by_subject(mimic3_path, table, output_path, items_to_keep=items_to_keep,
                                                subjects_to_keep=subjects)
