from process_files.extract_subjects import extract_subjects
from process_files.validate_events import validate_events
from process_files.extract_episodes_from_subjects import extract_episodes_from_subjects
from process_files.split_train_test_val import split_train_test_val
from process_files.create_in_hospital_mortality import create_in_hospital_mortality
import sys
import os

if __name__ == "__main__":
    mimic3_path = sys.argv[1]
    # # output_path = sys.argv[2]

    if not os.path.exists("./data/"):
        os.makedirs("./data")
    if not os.path.exists("./data/root/"):
        os.makedirs("./data/root/")

    print("EXTRACTING SUBJECTS")
    extract_subjects(mimic3_path)
    print()

    print("REMOVING INVALID EVENTS")
    validate_events()
    print()

    print("EXTRACTING EPISODES FROM SUBJECTS")
    extract_episodes_from_subjects()
    print()

    print("SPLITTING TRAINING, TESTING AND VALIDATION SET")
    split_train_test_val()
    print()

    print("CREATING FINAL DATASET")
    create_in_hospital_mortality()

    

