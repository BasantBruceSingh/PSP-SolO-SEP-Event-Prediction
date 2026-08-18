# PSP-SolO-SEP-Event-Prediction
This repository contains the data collection, machine learning, and plotting code to go with the paper "Solar Energetic Particle Prediction in the Inner Heliosphere Using Deep Learning, PSP/IS⊙IS,3 SolO/EPD and SHARP Data"

All code was run on Princeton University's Stellar research computing cluster.

Many of the raw CSV files were too large to be stored here. As a result, only the final data CSV files are stored, and intermediate ones are not.

This study contains two models: DeepSets-MLP and Attention-MLP. Both are binary classification models to whether or not J_{linlin} is above the threshold outlined in the paper.

## Prerequisites
Due to the lightweight nature of the SHARP data, all the data needed to train and evaluate the models is contained within this repository.

In order to convert the data into the necessary .npy files, run the files data_collection/combined_point_split_generation.py, data_collection/combined_point_split_generation_no_aggregation.py, and data_collection/combined_point_split_generation_na_kfold.py

## Running the models

This repository includes training scripts for two pointwise SEP classification models:

1. **Attention-MLP Classifier**
2. **DeepSets-MLP Classifier**

Both models operate on sets of active-region SHARP features at each timestamp. The models take as input a variable number of active regions, apply a permutation-invariant set encoder, and output a binary SEP event prediction.

The default command-line arguments correspond to the hyperparameters used in the study.

---

## Attention-MLP Classifier

The Attention-MLP model first projects each active-region feature vector into a learned embedding space, applies a Transformer encoder over the active-region set, mean-pools over real/non-padded active regions, and then applies an MLP classifier.

### Command-line arguments

| Argument | Default | Purpose |
|---|---:|---|
| `--epochs` | `20` | Number of training epochs. |
| `--learning_rate` | `1e-5` | Learning rate used by the AdamW optimizer. |
| `--threshold` | `0.5` | Probability threshold used to convert sigmoid outputs into binary predictions. |
| `--d_model` | `32` | Embedding dimension used inside the attention encoder. |
| `--n_heads` | `4` | Number of attention heads in the Transformer encoder layer. |
| `--num_layers` | `1` | Number of Transformer encoder layers. |
| `--mlp_hidden` | `16` | Number of hidden units in the final MLP classifier. |
| `--dropout` | `0.54` | Dropout probability used in the attention encoder and MLP classifier. |
| `--weight_decay` | `0.02` | Weight decay used by the AdamW optimizer. |

### Example usage

Run the Attention-MLP model with the default study parameters:

```bash
python attention_MLP.py
```

Run for 100 epochs:

```bash
python attention_MLP.py --epochs 100
```

Change the learning rate and dropout:

```bash
python attention_MLP.py --learning_rate 5e-5 --dropout 0.4
```

Change the Transformer encoder size:

```bash
python attention_MLP.py --d_model 64 --n_heads 4 --num_layers 2
```

Change the classification threshold:

```bash
python attention_MLP.py --threshold 0.4
```

---

## DeepSets-MLP Classifier

The DeepSets-MLP model encodes each active region independently using a small MLP, averages the active-region embeddings, applies a second MLP to produce a set embedding, and then classifies the resulting set representation.

The DeepSets encoder follows the form:

```text
e = rho( mean_i phi(x_i) )
```

where `phi` embeds each active-region feature vector, the mean operation aggregates across active regions, and `rho` maps the pooled representation to a set embedding.

### Command-line arguments

| Argument | Default | Purpose |
|---|---:|---|
| `--epochs` | `100` | Number of training epochs. |
| `--learning_rate` | `6e-4` | Learning rate used by the AdamW optimizer. |
| `--threshold` | `0.5` | Probability threshold used to convert sigmoid outputs into binary predictions. |
| `--mlp_hidden` | `16` | Number of hidden units in the final MLP classifier. |
| `--dropout` | `0.4` | Dropout probability used in the DeepSets encoder and MLP classifier. |
| `--individual_embedding_dim` | `32` | Dimension of the per-active-region embedding produced by `phi`. |
| `--set_embedding_dim` | `16` | Dimension of the final set embedding produced by `rho`. |

### Example usage

Run the DeepSets-MLP model with the default study parameters:

```bash
python deepsets_MLP.py
```

Run for 50 epochs:

```bash
python deepsets_MLP.py --epochs 50
```

Change the learning rate:

```bash
python deepsets_MLP.py --learning_rate 1e-4
```

Change the DeepSets embedding dimensions:

```bash
python deepsets_MLP.py --individual_embedding_dim 64 --set_embedding_dim 32
```

Change the dropout rate and classification threshold:

```bash
python deepsets_MLP.py --dropout 0.5 --threshold 0.4
```

---

## Training outputs

During training, the scripts call the shared `train(...)` function from `train_classification.py`. This function logs training and validation metrics, including:

| Metric | Description |
|---|---|
| Training loss | Binary cross-entropy loss with logits. |
| Training accuracy | Classification accuracy on the training set. |
| Validation accuracy | Classification accuracy on the validation set. |
| Precision | Fraction of predicted SEP events that are true SEP events. |
| Recall / POD | Probability of detection for SEP events. |
| FAR | False alarm ratio. |
| TSS | True skill statistic. |
| HSS | Heidke skill score. |
| F1 | Harmonic mean of precision and recall. |

The training script saves model weights in the `model_weights/` directory. The best validation-TSS checkpoint is saved as:

```text
model_weights/<model_name>_best.pth
```

The final checkpoint after the last epoch is saved as:

```text
model_weights/<model_name>_end.pth
```

---

## Notes

The default hyperparameters in the command-line arguments are the parameters used in the study. To reproduce the reported baseline training runs, run each script without overriding the defaults.

The scripts expect preprocessed train, validation, and test dataloaders to be available through `load_data_pointwise.py`.
