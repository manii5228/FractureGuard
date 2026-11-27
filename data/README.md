# Data directory

Dataset files are not committed to Git.

```text
data/
|-- raw/          # untouched extracted official releases
|-- derived/      # generated 8-bit images, converted labels, caches
|-- cache/        # temporary downloads and API responses
|-- manifests/    # inventory, normalized manifests, frozen splits
`-- README.md
```

Rules:

1. Download only from official dataset repositories.
2. Preserve original archives and raw extracted files unchanged.
3. Never place one patient's images in multiple splits.
4. Record file hashes and source versions.
5. Do not redistribute dataset files through this repository.
6. Do not attempt to identify any patient.

## Kaggle training bundle

The experiment uses the frozen train and validation cohort and fracture labels. Obtain both
inputs from the project team and place them at these paths in your local checkout.

- `data/manifests/grazpedwri_m2_cohort.csv`
- `data/manifests/grazpedwri_m2_labels.csv`

From the repository root, run:

```sh
python scripts/build_kaggle_bundle.py
```

Attach the resulting `work/kaggle/fractureguard_m2_bundle.zip` and the original 16-bit
GRAZPEDWRI-DX images as separate inputs to the
[Kaggle training notebook](../notebooks/kaggle_m2_training.ipynb).
