from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _attribute_name(value: str) -> str:
    return value.removeprefix("# ").strip()


def _read_attribute_csv(path: Path, *, has_image_column: bool) -> tuple[tuple[str, ...], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc

        offset = 1 if has_image_column else 0
        expected_prefix = "# image" if has_image_column else "# "
        if not header or not header[0].startswith(expected_prefix):
            raise ValueError(f"unexpected header in {path}: {header[:2]}")
        attribute_names = tuple(_attribute_name(value) for value in header[offset:])
        if len(attribute_names) != 40 or len(set(attribute_names)) != 40:
            raise ValueError(f"expected 40 unique attributes in {path}, got {len(attribute_names)}")

        image_paths: list[str] = []
        values: list[list[int]] = []
        expected_columns = 40 + offset
        for line_number, row in enumerate(rows, start=2):
            if len(row) != expected_columns:
                raise ValueError(f"{path}:{line_number}: expected {expected_columns} columns, got {len(row)}")
            if has_image_column:
                image_paths.append(row[0])
            try:
                attrs = [int(value) for value in row[offset:]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: attributes must be integers") from exc
            if any(value not in (0, 1) for value in attrs):
                raise ValueError(f"{path}:{line_number}: attributes must be binary")
            values.append(attrs)

    return attribute_names, image_paths, np.asarray(values, dtype=np.uint8)


@dataclass(frozen=True)
class RetrievalSplit:
    attribute_names: tuple[str, ...]
    image_paths: tuple[str, ...]
    gallery_attributes: np.ndarray
    gallery_query_ids: np.ndarray
    queries: np.ndarray

    @property
    def gallery_size(self) -> int:
        return int(self.gallery_attributes.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.queries.shape[0])

    @property
    def sources(self) -> np.ndarray:
        return np.asarray([path.split("/", 1)[0] for path in self.image_paths], dtype=str)

    def subset_gallery(self, mask: np.ndarray) -> "RetrievalSplit":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.gallery_size,):
            raise ValueError("gallery mask has the wrong shape")
        if not mask.any():
            raise ValueError("gallery subset is empty")
        return RetrievalSplit(
            attribute_names=self.attribute_names,
            image_paths=tuple(np.asarray(self.image_paths, dtype=str)[mask].tolist()),
            gallery_attributes=self.gallery_attributes[mask],
            gallery_query_ids=self.gallery_query_ids[mask],
            queries=self.queries,
        )


def load_retrieval_split(split_dir: str | Path) -> RetrievalSplit:
    split_dir = Path(split_dir)
    gt_path = split_dir / "gt.csv"
    ids_path = split_dir / "ids.csv"
    queries_path = split_dir / "queries.csv"

    gt_names, image_paths, gallery_attributes = _read_attribute_csv(gt_path, has_image_column=True)
    query_names, _, queries = _read_attribute_csv(queries_path, has_image_column=False)
    if gt_names != query_names:
        raise ValueError("gt.csv and queries.csv attribute order differs")

    id_paths: list[str] = []
    query_ids: list[int] = []
    with ids_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 2:
                raise ValueError(f"{ids_path}:{line_number}: expected path and query id")
            id_paths.append(row[0])
            try:
                query_ids.append(int(row[1]))
            except ValueError as exc:
                raise ValueError(f"{ids_path}:{line_number}: query id must be an integer") from exc

    gallery_query_ids = np.asarray(query_ids, dtype=np.int64)
    if tuple(image_paths) != tuple(id_paths):
        raise ValueError("gt.csv and ids.csv image order differs")
    if gallery_query_ids.shape != (gallery_attributes.shape[0],):
        raise ValueError("ids.csv row count differs from gt.csv")
    if queries.ndim != 2 or queries.shape[1] != 40 or queries.shape[0] == 0:
        raise ValueError("queries.csv must be a non-empty Q x 40 matrix")
    if gallery_query_ids.size == 0 or gallery_query_ids.min() < 0 or gallery_query_ids.max() >= queries.shape[0]:
        raise ValueError("ids.csv contains an out-of-range query id")
    if np.unique(queries, axis=0).shape[0] != queries.shape[0]:
        raise ValueError("queries.csv contains duplicate attribute vectors")

    counts = np.bincount(gallery_query_ids, minlength=queries.shape[0])
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)[:10].tolist()
        raise ValueError(f"queries without gallery matches: {missing}")
    expected = queries[gallery_query_ids]
    if not np.array_equal(expected, gallery_attributes):
        mismatched = np.flatnonzero(np.any(expected != gallery_attributes, axis=1))[:10].tolist()
        raise ValueError(f"ids.csv does not match the 40-bit gallery labels at rows {mismatched}")

    return RetrievalSplit(
        attribute_names=gt_names,
        image_paths=tuple(image_paths),
        gallery_attributes=gallery_attributes,
        gallery_query_ids=gallery_query_ids,
        queries=queries,
    )
