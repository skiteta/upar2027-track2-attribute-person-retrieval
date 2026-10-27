from pathlib import Path

import numpy as np

from upar.data import load_retrieval_split
from upar.submission import load_submission_module, make_fixture, validate_output


SAMPLE_DIR = Path("starter_kit/examples/task2/sample_code_submission")
VAL_DIR = Path("starter_kit/data/annotations/task2/val")


def test_official_sample_submission_contract() -> None:
    split = load_retrieval_split(VAL_DIR)
    fixture = make_fixture(split, query_count=3, gallery_size=7, image_root=Path("starter_kit/data"))
    module = load_submission_module(SAMPLE_DIR / "run.py")
    module.load_model()
    output = module.rank_gallery(fixture)
    kind, values = validate_output(output, query_count=3, gallery_size=7)
    assert kind == "distances"
    assert values.shape == (3, 7)
    assert np.isfinite(values).all()
    # The official no-pixel prior repeats one probability vector, hence every
    # image ties for a fixed query and Codabench preserves gallery order.
    assert np.all(values == values[:, :1])
