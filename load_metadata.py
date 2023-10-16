import os
from pathlib import Path
import json
import pandas as pd

def load_all_metadata(folderpath: str | Path) -> dict:
    """Load all metadata"""
    expected_keys = ['Run', 'Link', 'Author', 'Species', 'BSource', 'BType', 'Longitudinal', 'Subject', 'Vaccine', 'Disease', 'Age', 'Chain', 'Unique sequences', 'Total sequences', 'Isotype']
    
    all_metadata = []
    
    for filename in os.listdir(folderpath)[2:]:
        
        if not filename.endswith(".csv"):
            continue
        
        filepath = f"{folderpath}/{filename}"
        with open(filepath) as file:
            metadata = file.readline()
        metadata_dict = json.loads(metadata.replace('""', '"')[1:-2])
        assert list(metadata_dict.keys()) == expected_keys
        metadata_dict["filepath"] = filepath
        all_metadata.append(metadata_dict)

    return pd.DataFrame(all_metadata)

metadata_all = load_all_metadata("Gupta_2017")

from data import load_data_from_file
import numpy as np

def get_filenames(subject_id, isotype='IGHG'):
    
    metadata_all = pd.read_csv("metadata_all.csv")
    list_subjects = metadata_all['Subject'].unique()
    n_subjects = len(list_subjects)
    
    if subject_id not in list(range(n_subjects)):
        raise AssertionError(f"Please provide a valid subject_id number to function get_data_subject (between 0 and {n_subjects - 1}, because there are only {n_subjects} subjects)")
        
    subject_name = list_subjects[subject_id]
    
    metadata_subject = metadata_all[metadata_all["Subject"] == subject_name]
    
    # display(metadata_subject)
    filepaths_subject = metadata_subject['filepath'].tolist()
    if isotype is not None:
        metadata_out = metadata_subject[metadata_subject["Isotype"] == isotype]
    else: 
        metadata_out = metadata_subject
    return metadata_out

def get_data_subject(subject_id):
    
    metadata_all = pd.read_csv("metadata_all.csv")
    list_subjects = metadata_all['Subject'].unique()
    n_subjects = len(list_subjects)
    
    if subject_id not in list(range(n_subjects)):
        raise AssertionError(f"Please provide a valid subject_id number to function get_data_subject (between 0 and {n_subjects - 1}, because there are only {n_subjects} subjects)")
        
    subject_name = list_subjects[subject_id]
    print(f"Loading subject '{subject_name}'...")
    
    metadata_subject = metadata_all[metadata_all["Subject"] == subject_name]
    
    # display(metadata_subject)
    filepaths_subject = metadata_subject['filepath'].tolist()
    print(f"{len(filepaths_subject)} files")
    
    df_all = []
    for filepath in filepaths_subject[:2]:
        
        df, metadata = load_data_from_file(filepath)
        # print(df.columns)
        df_all.append(df)
    
    expected_columns = df_all[0].columns
    for df, filepath in zip(df_all, filepaths_subject):
        # print(expected_columns, df.columns)
        if list(df.columns) != list(expected_columns):
            raise AssertionError(f"{filepath} doesn't have the expected columns; got {list(df.columns)}, expected {list(expected_columns)}")
    print("Loaded!")

    print(len(df_all), np.sum([len(df) for df in df_all]))
    return pd.concat(df_all)


get_data_subject(2)
