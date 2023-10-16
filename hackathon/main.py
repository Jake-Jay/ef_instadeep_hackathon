from __future__ import annotations

from pathlib import Path

import seaborn as sns

from transformers import AutoTokenizer
from data import TokenisedBCR

filepath = Path("Gupta_2017/SRR4431764_1_Heavy_Bulk.csv")

tokeniser = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")
tokenised_dataset = TokenisedBCR(filepaths=[filepath], tokeniser=tokeniser)