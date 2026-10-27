from pathlib import Path

import numpy as np

from upar.data import load_retrieval_split


VAL_DIR = Path("starter_kit/data/annotations/task2/val")


def test_official_validation_contract() -> None:
    split = load_retrieval_split(VAL_DIR)
    assert split.gallery_size == 33_407
    assert split.query_count == 3_462
    assert split.gallery_attributes.shape == (33_407, 40)
    assert split.queries.shape == (3_462, 40)
    assert set(np.unique(split.sources)) == {"Market1501", "PA100k", "PETA"}
    assert np.array_equal(split.queries[split.gallery_query_ids], split.gallery_attributes)
