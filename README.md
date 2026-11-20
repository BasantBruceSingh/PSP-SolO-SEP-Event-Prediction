# PSP-SolO-SEP-Event-Prediction

This repository contains the data collection, machine learning, and plotting
code for *Solar Energetic Particle Prediction in the Inner Heliosphere Using
Deep Learning, PSP/IS⊙IS,3 SolO/EPD and SHARP Data*. The study runs used
Princeton University's Stellar research computing cluster.

The two classifiers, **DeepSets-MLP** and **Attention-MLP**, use Solar Dynamics
Observatory SHARP features to predict whether the particle flux target
`J_linlin` exceeds the cutoff used to define a positive sample.
The models work with observations from Parker Solar Probe (PSP), Solar
Orbiter (SolO), or both spacecraft pooled together.

## Code map

| Location | What to find there |
| --- | --- |
| [data_collection/](data_collection/) | Spacecraft data ingestion, timestamp and footprint mapping, alignment with SHARP, and split generation. |
| [set_classifiers.py](initial_models/set_classifiers.py) | Reusable PyTorch implementations of the two set encoders and classifier heads. |
| [load_data_pointwise.py](initial_models/load_data_pointwise.py) and [load_data_pointwise_kfold.py](initial_models/load_data_pointwise_kfold.py) | Load saved single-split and repeated-holdout arrays and construct binary labels. |
| [train_classification.py](initial_models/train_classification.py) | Weighted loss, training loop, checkpointing, and binary classification metrics. |
| [initial_models/](initial_models/) | Training, Optuna, and evaluation entry points, plus Slurm launchers and exploratory baselines. |
| [initial_plots/](initial_plots/) | Event and flux plots, data diagnostics, and training-curve plots. |

## Setup and inputs

The Slurm launchers use the Conda environment `sepprediction311` and request
a GPU. For a local run, use Python 3.11 with PyTorch, NumPy, pandas,
scikit-learn, Optuna, and Weights & Biases. Install `openpyxl` to read the SolO
event catalogue. The main training scripts select CUDA when available.
The Slurm launchers contain Stellar-specific paths and module settings.

The repository tracks the final aligned observation tables
`data_collection/SHARP_to_PSP_times.csv` and
`data_collection/SHARP_to_SolO_times.csv`. Large raw and intermediate files
from the upstream alignment workflow are omitted. To regenerate the arrays,
also place `psp_isois_eventlist_v21.csv` and `SolO_catalogue.xlsx` in
`data_collection/`. Those catalogues are present in the study workspace and
are currently untracked. If you already have the generated arrays, you can go
straight to training.

## Generate the model inputs

Run the generators from `data_collection/`. Each writes to its local
`npy/` directory.

| Data needed | Generator |
| --- | --- |
| Combined PSP and SolO, one train/validation/test split | [combined_point_generator_no_aggregation.py](data_collection/combined_point_generator_no_aggregation.py) |
| PSP and SolO separately, one split each | [separate_point_generator_na.py](data_collection/separate_point_generator_na.py) |
| Combined PSP and SolO, ten repeated holdouts | [combined_point_generator_na_kfold.py](data_collection/combined_point_generator_na_kfold.py) |
| PSP and SolO separately, ten repeated holdouts each | [separate_point_generator_na_kfold.py](data_collection/separate_point_generator_na_kfold.py) |
| Aggregated SHARP features for each spacecraft | [combined_point_split_generation.py](data_collection/combined_point_split_generation.py) |

For example, from the repository root, generate the combined single-split
data with `(cd data_collection && python combined_point_generator_no_aggregation.py)`.
The aggregated workflow produces per-timestamp feature vectors. The two set
classifiers use the generators that retain individual active regions.

The set-model loaders read aligned `X_t_*`, `mask_t_*`, and `y_t_*` arrays.
A batch has feature shape `[batch, active_regions, features]` and mask shape
`[batch, active_regions]`. A true mask entry marks a real active region.
The default pointwise models use 10 SHARP features per region and emit one
logit per timestamp.

The loaders mark an observation positive when its standardized log flux is
strictly greater than **1.0**. In the repeated-holdout generators, the target
normalization is fitted to each spacecraft's training data within the
holdout, so the equivalent raw-flux cutoff can vary across spacecraft and
holdouts. The CLI flag `--threshold` controls the **prediction probability**
cutoff applied after the sigmoid. Changing it leaves the dataset labels
unchanged.

## Train the single-split models

From the repository root, run the following commands with the prepared arrays
in `data_collection/npy/`.

```bash
cd initial_models
mkdir -p model_weights
python attention_MLP_classification.py
python deepsets_MLP_classification.py
```

The default dataset pools both spacecraft. Pass `--psp_only` or `--solo_only`
to select one spacecraft. Use `--help` for the full argument list. The
defaults below are those in the current scripts. Published run settings
should be matched to the corresponding study configuration.

### Attention-MLP

Each region's features are projected into an embedding. A Transformer
encoder lets regions attend to one another, then masked mean pooling
summarizes the valid regions. An MLP maps that summary to a logit.

| Argument | Default | Controls |
| --- | ---: | --- |
| `--epochs` | 100 | Training epochs |
| `--learning_rate` | `1e-4` | AdamW learning rate |
| `--threshold` | `0.5` | Prediction probability cutoff |
| `--d_model` | 32 | Transformer embedding width |
| `--n_heads` | 4 | Number of attention heads. The embedding width must be divisible by this value. |
| `--num_layers` | 1 | Transformer encoder layers |
| `--mlp_hidden` | 16 | Classifier hidden width |
| `--dropout` | `0.54` | Encoder and classifier dropout |
| `--weight_decay` | `0.02` | AdamW weight decay |

```bash
python attention_MLP_classification.py --learning_rate 5e-5 --dropout 0.4
python attention_MLP_classification.py --d_model 64 --n_heads 4 --num_layers 2
```

### DeepSets-MLP

A shared MLP `phi` embeds each active region independently. The encoder
averages the real-region embeddings and applies a second MLP `rho` to get
the set embedding. The classifier maps it to a logit.

```text
set_embedding = rho(mean(phi(x_i) for real active regions i))
```

| Argument | Default | Controls |
| --- | ---: | --- |
| `--epochs` | 100 | Training epochs |
| `--learning_rate` | `5e-5` | AdamW learning rate |
| `--threshold` | `0.5` | Prediction probability cutoff |
| `--mlp_hidden` | 16 | Classifier hidden width |
| `--dropout` | `0.4` | Encoder and classifier dropout. The current CLI parses explicit overrides as integers. |
| `--individual_embedding_dim` | 32 | Width of the per-region embedding |
| `--set_embedding_dim` | 16 | Width of the pooled set embedding |
| `--weight_decay` | `0.0` | AdamW weight decay |

```bash
python deepsets_MLP_classification.py --epochs 50
python deepsets_MLP_classification.py --individual_embedding_dim 64 --set_embedding_dim 32
```

## Repeated holdouts and evaluation

The `kfold` scripts use ten seeded temporal holdouts, indexed from 0 to 9.
Within a holdout, overlapping SEP event windows are merged and assigned as
whole blocks. Quiet-time observations form their own blocks. The generators
allocate roughly 70% of each block type to training, 15% to validation, and
the remainder to test. Block sizes and class proportions can differ across
splits, and blocks can move between splits across repeats. The saved
`shared_time_split*.csv` files record those assignments.

The [Attention launcher](initial_models/attention_MLP_classification_kfold.sh)
and [DeepSets launcher](initial_models/deepsets_MLP_classification_kfold.sh)
train the combined, PSP-only, and SolO-only datasets with the checkpoint names
the evaluators expect. The Attention holdout trainer uses the
`sep_attention_tss_v2` study in `optuna_sep_attention_tss_v2.db` to select
the model architecture. The separate `*_tuning_kfold.py` scripts run
hyperparameter searches with their own studies.

After training, run the following from `initial_models/` with the matching
arrays and checkpoints.

```bash
python attention_eval_kfold.py
python deepsets_eval_kfold.py
```

Each evaluator loads the final-epoch checkpoint for each dataset and holdout
and applies the same probability cutoff to its validation and test splits.
The summaries report the mean and standard deviation across holdouts.
Preserve the generated arrays and the Attention study alongside checkpoints
so later evaluation uses the same splits and architecture.

## Training outputs and metrics

The shared trainer uses binary cross-entropy with logits. The single-split
entry points set the positive-class weight from the training label counts.
Training progress goes to the console and, when enabled, to offline Weights
& Biases logs.

| Checkpoint | When it is written |
| --- | --- |
| `model_weights/<model_name>_best.pth` | When validation TSS improves |
| `model_weights/<model_name>_end.pth` | At the end of training |

The shared evaluator builds a confusion matrix across the full split before
computing the scores.

| Metric | Meaning |
| --- | --- |
| Accuracy | Fraction of samples classified correctly |
| Precision | Fraction of predicted positive samples that are positive |
| Recall / POD | Fraction of positive samples detected |
| FAR | Fraction of predicted positive samples that are false alarms |
| TSS | True-positive rate minus false-positive rate |
| HSS | Heidke skill score, which adjusts for chance agreement |
| F1 | Harmonic mean of precision and recall |
