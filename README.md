# TV-Survival

This repository contains the implementation and reproducibility materials for the experimental results.

## Reproducibility Guide

To reproduce the results, follow these steps in order:

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

Execute the three Jupyter notebooks in the `demo/` folder **sequentially**. Each notebook contains detailed explanations and parameter configurations required for reproducing the results:

#### 1. [Data Preprocessing](demo/1.%20preprocessing_demo.ipynb)

This notebook:
- Downloads and loads the raw dataset
- Converts data to Parquet format using `datastats2parquet.py`
- Applies truncation procedures with `truncate.py`
- Splits data into train/validation/test sets using `Splitter.py`

#### 2. [Model Training](demo/2.%20training_demo.ipynb)

This notebook:
- Configures training parameters for the experiments
- Trains models with time-varying covariates
- Saves trained models for aggregation step

#### 3. [Model Aggregation](demo/3.%20aggregation_demo.ipynb)

This notebook:
- Loads pre-trained models from step 2
- Implements various aggregation strategies (`n_dist`, `t_dist`, `geom`, `prob_dist`)
- Evaluates aggregation performance with different sample sizes
- Generates the final results

## Data Source

The experiments use hard drive failure data from Backblaze, a cloud storage provider that publicly releases their drive statistics. The preprocessing notebook handles data download and preparation.