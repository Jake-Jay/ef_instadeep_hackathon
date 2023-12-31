from __future__ import annotations

from pathlib import Path

import pandas as pd
import json




# model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")
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
    if metadata["Longitudinal"] == "After-Week-2":
        return 1
    elif metadata["Longitudinal"] == "blahblah":
        return -1
    else:
        raise ValueError

def load_data_from_file(filepath: str | Path) -> tuple(pd.DataFrame, dict):
    """Load a single entry from OAS"""
    with open(filepath) as file:
        metadata = file.readline()
    metadata_dict = json.loads(metadata.replace('""', '"')[1:-2])
    
    df = pd.read_csv(filepath, skiprows=[0])
    df["label"] = get_label(metadata_dict)
    
    return df, metadata_dict

# ------------------------------------------------------------
# Data loading classes
# ------------------------------------------------------------

class BCR:
    """Wrap a B Cell Repetoire"""
    
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
    
    def __getitem__(self, idx: int) -> tuple(str, str):
        return tuple(self.data.iloc[0][["sequence", "label"]])
          
        
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

