from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hackathon.data import BCR

if __name__ == "__main__":

    filepath = Path("data/SRR4431764_1_Heavy_Bulk.csv")
    bcr = BCR(
        filepaths=[filepath],
        cache_dirpath=None,
    )

    
