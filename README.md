# FractureGuard

FractureGuard is a university research prototype for detecting possible fractures in pediatric wrist X-rays. It processes frontal and lateral views on a CPU, marks candidate fracture regions, and combines the two views into a study-level suggestion. A reviewer checks the images, writes the findings, and signs and releases a PDF report.

Use de-identified images for research and demonstrations. The prototype is for coursework and research use.

## Start here

| I want to… | Go to |
| --- | --- |
| See the upload, review, and report workflow | [Demo video](reports/FractureGuard_Milestone_7_Demo.mp4) |
| Run the website and try a public X-ray pair | [Local demo instructions](reports/M7_VERIFICATION.md) |
| Read the paper or edit the LaTeX | [Six-page PDF](reports/m8/FractureGuard.pdf) · [LaTeX and figures ZIP](reports/m8/FractureGuard_M8_Source.zip) |
| Reproduce the reported tables from the saved results | [Paper reproduction guide](reports/m8/REPRODUCE.md) |
| See the study protocol and milestone record | [Project plan](FRACTUREGUARD_PROJECT_PLAN.md) |

## Run the website locally

Use Python 3.11. In PowerShell, from the project folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,ml]"
python -m fractureguard.web migrate --noinput
python -m fractureguard.web bootstrap_admin operator
```

In two terminals with the same environment active, run the worker in one and the web server in the other:

```powershell
python -m fractureguard.web runworker
```

```powershell
python -m fractureguard.web serve --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Use the administrator account to invite a demo user and a reviewer. The demo user submits two PNGs of the same wrist. The reviewer checks both views, corrects any boxes, enters findings, signs the report, and releases it. The demo user can then download the PDF. The [demo instructions](reports/M7_VERIFICATION.md) list public image pairs and the exact account steps.

The worker uses the saved [FP32 ONNX model](reports/models/m6/yolo26n_fp32.onnx) and [frozen triage policy](reports/predictions/m4/m4_frozen_triage_policy.json). You can run the demo without retraining.

## Data and training

| Dataset | Role |
| --- | --- |
| [GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193.v2) | The project uses the public fracture annotations for training, validation, and internal testing. Patients are assigned to one split only. |
| [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) | Candidate for external testing. That evaluation is pending independent verification of wrist-compatible labels. |

The frozen split and label tables are in [data/manifests](data/manifests). Original downloads live in `data/raw` and account for about 15.5 GiB of this local workspace. The installed Python environment takes about 1.7 GiB. The training pipeline rebuilds intermediate images under `data/derived` from the original radiographs and frozen manifests.

FractureGuard's [training runner](scripts/run_m2_pipeline.py) prepares the data and configures the runs. It fine-tunes Ultralytics pretrained detectors through the Ultralytics training API. To run the Kaggle workflow, build the bundle and attach it alongside GRAZPEDWRI-DX to the [training notebook](notebooks/kaggle_m2_training.ipynb).

```powershell
python scripts/build_kaggle_bundle.py
```

The bundle is written to `work/kaggle/fractureguard_m2_bundle.zip`. The notebook explains how to resume stages. The [paper reproduction guide](reports/m8/REPRODUCE.md#training-code-and-model-provenance) documents the differences between the planned protocol, archived runs, and retained checkpoint.

## Measured results

| Measure | Internal evaluation |
| --- | ---: |
| Held-out studies | 826 |
| Study AUROC | 0.9793 |
| Accuracy among 771 definitive suggestions | 97.02% |
| Studies deferred as uncertain | 55 |
| Annotated fractures among 411 no-fracture suggestions | 20 |
| Median time for a paired study on the benchmark CPU | 97.6 ms |

The findings come from one dataset and one benchmark CPU. The website workflow was tested locally. Clinical use and external performance remain unevaluated.

## Project layout

| Folder | Contents |
| --- | --- |
| [src/fractureguard](src/fractureguard) | Research pipeline and Django application |
| [configs](configs), [scripts](scripts), [notebooks](notebooks) | Experiment settings and commands |
| [data/manifests](data/manifests) | Dataset audit, splits, and labels |
| [reports](reports) | Saved predictions, model exports, demo, and paper |
| [tests](tests) | Project tests |

For development, install `python -m pip install -e ".[dev]"` and run `python -m pytest`. See [LICENSE](LICENSE) for the project license notice and the dataset pages for their terms.

