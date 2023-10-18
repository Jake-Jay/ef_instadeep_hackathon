from __future__ import annotations

from pathlib import Path

import pandas as pd
import json

DIR = 'Gupta_2017' 
FILE = 'Gupta_2017/SRR4431764_1_Heavy_Bulk.csv' 
LONGITUDINALS = ['Before-Hour-1', 'After-Week-2']

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_label(metadata: dict) -> str:
    if metadata["Longitudinal"] == LONGITUDINALS[1]:
        return 1
    elif metadata["Longitudinal"] == LONGITUDINALS[0]:
        return -1
    else:
        return 0

def load_data_from_file(filepath: str | Path) -> tuple(pd.DataFrame, dict):
    """Load a single entry from OAS"""
    
    with open(filepath) as file:
        metadata = file.readline()
    metadata_dict = json.loads(metadata.replace('""', '"')[1:-2])
    
    df = pd.read_csv(filepath, skiprows=[0])
    df["label"] = get_label(metadata_dict)
    
    return df, metadata_dict


def filter_metadata(
    subject_id : int,
    isotype: str ='IGHG',
    longitudinals = LONGITUDINALS,
) -> pd.DataFrame:
    """ Return filtered dataframe by subject, isotype and longitudinals """    
    metadata_all = pd.read_csv("metadata_all.csv")
    list_subjects = metadata_all['Subject'].unique()
    n_subjects = len(list_subjects)
    
    if subject_id not in list(range(n_subjects)):
        raise AssertionError(
            f"Please provide a valid subject_id number to function "
            f"get_data_subject (between 0 and {n_subjects - 1}, "
            f"because there are only {n_subjects} subjects)"
        )
        
    subject_name = list_subjects[subject_id]
    
    metadata_subject = metadata_all[metadata_all["Subject"] == subject_name]
    
    # display(metadata_subject)
    filepaths_subject = metadata_subject['filepath'].tolist()
    if isotype is not None:
        metadata_out = metadata_subject[metadata_subject["Isotype"] == isotype]
    else: 
        metadata_out = metadata_subject
    
    if longitudinals is not None:
        metadata_out = metadata_out[metadata_out["Longitudinal"].isin(longitudinals)]
        print(metadata_out["Longitudinal"].unique())
    return metadata_out

def get_filenames(*args, **kwargs):
    """ Return filenames filtered by subject, isotype and longitudinals """
    return filter_metadata(*args, **kwargs)['filepath'].tolist()

# ------------------------------------------------------------
# Data loading classes
# ------------------------------------------------------------

class BCR:
    """Wrap a B Cell Repertoire"""
    
    def __init__(
        self,
        filepaths: list[Path | str],
        cache_dirpath: Path | str | None = None,
    ) -> None:
        
        self.filepaths = filepaths

        _all_data, self.metadata = zip(*[
            load_data_from_file(filepath)
            for filepath in self.filepaths
        ])

        self.data = pd.concat(_all_data)
        self.data["sequence_len"] = self.data["sequence"].apply(len)
        self.data = self.data.loc[
            self.data["sequence_len"] > 300   
        ]
    
    def sequences(self, n: int) -> list[str]:
        return self.data.sample(n, random_state=42)["sequence"]
    
    def __getitem__(self, idx: int) -> tuple(str, str):
        return tuple(self.data.iloc[idx][["sequence", "label"]])
    
    def __len__(self) -> int:
        return len(self.data)
          
        
class TokenisedBCR(BCR):
    """Tokenise a BCR and follow Torch Dataset conventions"""
    
    def __init__(
        self,
        tokeniser,        
        filepaths: list[Path | str],
        cache_dirpath: Path | str | None = None,
    ) -> None:
        super().__init__(filepaths, cache_dirpath)

        self.tokeniser = tokeniser

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with keys:
        - input_ids
        - attention_mask
        - label
        """
        sequence, label = super().__getitem__(idx)
        
        return {
            **self.tokeniser(sequence),
            "label": label,
        }

class EmbedBCR:
    """Use nuceotide transformer to get embeddings for a seq"""
    
    def __init__(
        self,
        tokenised_dataset: TokenisedDataset,
    ) -> None:
        pass
        
        
    def _embed(self) -> None:
        pass
        







    