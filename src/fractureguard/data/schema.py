from __future__ import annotations

MANIFEST_COLUMNS = [
    "dataset_name",
    "image_id",
    "patient_id",
    "study_id",
    "image_path",
    "projection",
    "laterality",
    "age",
    "sex",
    "initial_exam",
    "diagnosis_uncertain",
    "cast",
    "metal",
    "osteopenia",
    "fracture_visible",
    "fracture_box_count",
    "label_path",
    "split",
    "source_bit_depth",
    "width",
    "height",
    "file_size_bytes",
    "file_hash_sha256",
]


GENERIC_AUDIT_COLUMNS = [
    "dataset_name",
    "image_id",
    "image_path",
    "extension",
    "width",
    "height",
    "image_mode",
    "source_bit_depth",
    "file_size_bytes",
    "file_hash_sha256",
    "read_error",
]
