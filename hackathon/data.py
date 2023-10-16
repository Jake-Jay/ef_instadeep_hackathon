from __future__ import annotations

from pathlib import Path

import pandas as pd
import json
# import haiku as hk
# import jax
# import jax.numpy as jnp

# from nucleotide_transformer.pretrained import get_pretrained_model
# from nucleotide_transformer.tokenizers import FixedSizeNucleotidesKmersTokenizer


DIR = 'Gupta_2017' 
FILE = 'Gupta_2017/SRR4431764_1_Heavy_Bulk.csv' 

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_label(metadata: dict) -> str:
    if metadata["date"] == "blah":
        return -1
    elif metadata["date"] == "blahblah":
        return 1
    else:
        raise ValueError

def load_data_from_file(filepath: str | Path) -> tuple(pd.DataFrame, dict):
    """Load a single entry from OAS"""
    with open(filepath) as file:
        metadata = file.readline()
    metadata_dict = json.loads(metadata.replace('""', '"')[1:-2])
    
    df = pd.read_csv(filepath, skiprows=[0])
    df["label"] = get_label(metadata)
    
    return df, metadata_dict

# ------------------------------------------------------------
# Data loading classes
# ------------------------------------------------------------

class BCR:
    """Wrap a B Cell Repetoire"""
    
    def __init__(
        self,
        filepaths: list[Path | str],
        cache_dirpath: Path | str,
    ) -> None:
        
        self.filepaths = filepaths

        
        _all_data, _all_metadata = zip(*[
            load_data_from_file(filepath)
            for filepath in self.filepaths
        ])

        self.data = pd.concat(_all_data)
    
    
    def __getitem__(self, idx: int) -> tuple(str, str):
        return self.data.iloc[idx]["sequence", "label"]
          
        
class TokenisedBCR(BCR):
    """Tokenise a BCR and follow Torch Dataset conventions"""
    
    def __init__(
        self,
        filepath: Path | str,
        cache_dirpath: Path | str,
        tokeniser,
    ) -> None:
        super().__init__(filepath, cache_dirpath)

        self.tokeniser = tokeniser

    def __getitem__(self, idx: int) -> str:
        sequence = super().__getitem__(idx)









    