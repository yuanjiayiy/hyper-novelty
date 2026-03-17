import ast
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_citation import get_citation_s2_batch, counts_by_year_to_trajectory
from tqdm import tqdm
import datasets
from datasets import Dataset, DatasetDict, load_dataset



if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") is None:
        raise ValueError("HF_TOKEN environment variable is not set")
    dataset_id = "yuancarrieyjy/AI-arXiv"
    ds = datasets.load_dataset(dataset_id)
    
    for split in ds.keys():
        df = ds[split].to_pandas()
        print(f"Processing {split} dataset with {len(df)} rows")
        # Use paperID (S2 paperId hash) or corpus_id if available
        id_col = "arxiv_id"
        paper_ids = df[id_col].dropna().unique().tolist()
        paper_ids = [f"arXiv:{pid}" for pid in paper_ids]
        print(f"Found {len(paper_ids)} unique paper IDs")

        cid2total = {}
        cid2trajectory = {}

        for pid, total, by_year, pub_year in tqdm(
            get_citation_s2_batch(paper_ids, sleep_between=1.0),
            total=len(paper_ids),
            desc="Fetching citations",
        ):
            cid2total[pid] = total if total is not None else np.nan
            cid2trajectory[pid] = counts_by_year_to_trajectory(by_year, pub_year) if pub_year and by_year else []

        df["total_citation_count"] = df[id_col].map(lambda x: cid2total.get(x, np.nan))
        df["citation_trajectory"] = df[id_col].map(lambda x: cid2trajectory.get(x, []))
        ds[split] = Dataset.from_pandas(df)
    ds.push_to_hub(dataset_id, token=os.environ.get("HF_TOKEN"))