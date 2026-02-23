import os
import time
import copy
import pickle
import shutil
from dataclasses import dataclass, is_dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from scoring import ModelScorer
from utils.Dataset import DiskDataset
from models import SurvPredictor
from models import CoxTimeVaryingEstimator


# Configs
@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    early_stopping: bool
    patience: int
    min_delta: float
    score_metric: str


@dataclass
class DataConfig:
    data_folder: str
    train_batchsize: int
    score_batchsize: int
    times: np.ndarray
    train_times: np.ndarray
    to_cens_shift: List[int]
    to_term_shift: List[int]
    cens_prob: float


@dataclass
class ExperimentConfig:
    metrics: List[str]
    schema: Dict[str, list]
    res_filename: str
    models_folder: str
    log_dir: str


# Utils

def make_schema(metrics: List[str], hparams: List[str]) -> Dict[str, list]:
    schema = {
        'train_samples': None,
        'method': [],
        'model_id': None,
        'train_time': None,
        'test_time': None,
        'error': None,
        'error_text': None
    }
    for m in metrics:
        schema[f'{m}_train_same_size'] = None
        schema[f'{m}_train_max_size'] = None
        schema[f'{m}_test'] = None
    for hp in hparams:
        schema[hp] = None
    return schema


def write_dict(filename: str, d: Dict[str, list]):
    pd.DataFrame(d).to_csv(filename, mode='a', header=False, index=False)


def create_res_file(filename: str, schema: Dict[str, list]):
    pd.DataFrame(schema).to_csv(filename, index=False)


def save_model(model, model_id: str, folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, f"{model_id}.pkl"), "wb") as f:
        pickle.dump(model, f)


def prepare_dataloader(
    train_samples: int,
    data_cfg: DataConfig,
    data_type: str,
    dataset_type: str,
    test_samples: Optional[int] = None,
    data_ext: str = ".csv" # or ".parquet"
):
    if data_type == "train":
        files = [f"{data_cfg.data_folder}/{train_samples}_train_preprocessed{data_ext}"]
        batch_size = data_cfg.train_batchsize
    else:
        if test_samples is None:
            raise ValueError("test_samples must be provided for test loader")
        files = [f"{data_cfg.data_folder}/{train_samples}_{test_samples}_test_preprocessed{data_ext}"]
        batch_size = data_cfg.score_batchsize

    ds = DiskDataset(
        dataset_type,
        files,
        to_cens_time_list=data_cfg.to_cens_shift,
        to_term_time_list=data_cfg.to_term_shift,
        cens_prob=data_cfg.cens_prob,
    )
    return DataLoader(ds, batch_size=batch_size)


# models

def prepare_model(method: str, hparams: Dict[str, Any], epochs: int, lr: float):
    if method == "SP":
        return SurvPredictor(28, hidden_dim=hparams["hidden_dim"], epochs=epochs, lr=lr)
    if method == "CoxTV":
        return CoxTimeVaryingEstimator(**hparams)

    raise ValueError(f"Unknown method {method}")


def fit_model(
    model,
    method: str,
    dl_train,
    dl_val,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
    writer,
):
    if method == "SP":
        model.fit(
            dl_train,
            times=data_cfg.train_times,
            val_dataloader=dl_val,
            early_stopping=train_cfg.early_stopping,
            score_metric=train_cfg.score_metric,
            patience=train_cfg.patience,
            min_delta=train_cfg.min_delta,
            writer=writer,
        )

    elif method.startswith("Cox"):
        model.fit(dl_train)


# Experiment

def run_single_experiment(
    train_samples: int,
    test_grid: List[int],
    method: str,
    hparams: Dict[str, Any],
    run_id: int,
    scorer: ModelScorer,
    dl_score_max,
    df_train_max,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    val_data_size: int,
):
    stats = copy.deepcopy(exp_cfg.schema)
    stats["train_samples"] = [train_samples]
    stats["method"] = [method]
    stats["model_id"] = [f"{run_id}_{method}"]
    stats["error"] = [0]
    stats["error_text"] = [""]

    for k, v in hparams.items():
        stats[k] = [v]

    writer = SummaryWriter(os.path.join(exp_cfg.log_dir, f"{method}_{run_id}"))

    try:
        dl_train = prepare_dataloader(train_samples, data_cfg, "train", "train")
        dl_train_score = prepare_dataloader(train_samples, data_cfg, "train", "score")
        dl_val = prepare_dataloader(
            train_samples, data_cfg, "test", "score", val_data_size
        )

        model = prepare_model(method, hparams, train_cfg.epochs, train_cfg.lr)

        t0 = time.time()
        fit_model(model, method, dl_train, dl_val, train_cfg, data_cfg, writer)
        stats["train_time"] = [time.time() - t0]
        preds, gt = model.predict(dl_train_score, data_cfg.times)
        train_metrics = scorer.get_metrics(
            model, preds, gt, data_cfg.times,
            metrics=exp_cfg.metrics,
            df_train=df_train_max,
        )

        for m in exp_cfg.metrics:
            stats[f"{m}_train_same_size"] = [train_metrics[m]]

        preds, gt = model.predict(dl_score_max, data_cfg.times)
        max_metrics = scorer.get_metrics(
            model, preds, gt, data_cfg.times,
            metrics=exp_cfg.metrics,
            df_train=df_train_max,
        )

        for m in exp_cfg.metrics:
            stats[f"{m}_train_max_size"] = [max_metrics[m]]

        for ts in test_grid:
            dl_test = prepare_dataloader(train_samples, data_cfg, "test", "score", ts)
            t0 = time.time()
            preds, gt = model.predict(dl_test, data_cfg.times)
            test_metrics = scorer.get_metrics(
                model, preds, gt, data_cfg.times,
                metrics=exp_cfg.metrics,
                df_train=df_train_max,
            )
            stats["test_time"] = [time.time() - t0]
            for m in exp_cfg.metrics:
                stats[f"{m}_test"] = [test_metrics[m]]

        save_model(model, stats["model_id"][0], exp_cfg.models_folder)
        write_dict(exp_cfg.res_filename, stats)

    except Exception as e:
        stats["error"] = [1]
        stats["error_text"] = [str(e)]
        write_dict(exp_cfg.res_filename, stats)

    return stats


def dump_config(f, name: str, cfg):
    f.write(f"[{name}]\n")
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    for k, v in cfg.items():
        f.write(f"{k} = {v}\n")
    f.write("\n")


def collect_all_hparams(param_grids: Dict[str, ParameterGrid]) -> List[str]:
    hparams = set()
    for grid in param_grids.values():
        for params in grid:
            hparams.update(params.keys())
    return sorted(hparams)


def init_experiments_folder(
    exp_num: int,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    exp_desc: str,
    files_to_save: List[str],
):
    base_dir = os.path.join("Artifacts", f"Exp_{exp_num}")
    models_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    code_dir = os.path.join(base_dir, "code")
    res_file = os.path.join(base_dir, "grid_search.csv")
    desc_file = os.path.join(base_dir, "Description.txt")

    if os.path.exists(base_dir):
        raise ValueError(f"Experiment Exp_{exp_num} already exists")

    os.makedirs(models_dir)
    os.makedirs(log_dir)
    os.makedirs(code_dir)

    # CSV schema
    pd.DataFrame(exp_cfg.schema).to_csv(res_file, index=False)

    # Description.txt
    with open(desc_file, "w") as f:
        dump_config(f, "Experiment", {
            "exp_num": exp_num,
            "metrics": exp_cfg.metrics,
        })
        dump_config(f, "DataConfig", data_cfg)
        dump_config(f, "TrainingConfig", train_cfg)
        f.write("[Description]\n")
        f.write(exp_desc)

    # Copy code
    for file in files_to_save:
        shutil.copy(file, code_dir)

    return base_dir, models_dir, log_dir, res_file


# Эти переменные используются в другом файле, который ещё не изменён к новому формату
TIMES = np.arange(0, 730)
TRAIN_TIMES = TIMES[::10]
TRAIN_BATCHSIZE = 512

if __name__ == "__main__":
    np.random.seed(42)

    # Global config
    EXP_NUM = 105

    TRAIN_GRID = [1]
    TEST_GRID = [25]
    HIDDEN_DIM_GRID = [2048]

    VAL_DATA_SIZE = 10

    EXP_DESC = "Survival Analysis Experiments"
    FILES_TO_SAVE = ["Experiments.py",
                     "disk_analyzer/models/Net.py",
                     "disk_analyzer/models/Dataset.py",
                     "disk_analyzer/models/SurvPredictor.py",
                     "disk_analyzer/models/Cox.py",
                     ]

    PARAM_GRIDS = {
        "SP": ParameterGrid({
            "hidden_dim": [2048],
        }),

        # "CoxTV": ParameterGrid({
        #     "penalizer": [np.logspace(-2, 2, 5)[1]],
        #     "l1_ratio": [np.logspace(-2, 2, 5)[1]]
        # }),
    }
    METHODS = PARAM_GRIDS.keys()

    # Paths
    DATA_FOLDER = "Data/Preprocessed_new"
    BASE_DIR = f"Artifacts/Exp_{EXP_NUM}"
    RES_FILE = f"{BASE_DIR}/grid_search.csv"
    MODELS_DIR = f"{BASE_DIR}/models"
    LOG_DIR = f"{BASE_DIR}/logs"
    # Metrics
    METRICS = ["ci", "ibs", "ibs_bal", "iauc"]

    ALL_HPARAMS = collect_all_hparams(PARAM_GRIDS)
    SCHEMA = make_schema(METRICS, ALL_HPARAMS)

    # Configs
    data_cfg = DataConfig(
        data_folder=DATA_FOLDER,
        train_batchsize=TRAIN_BATCHSIZE,
        score_batchsize=512,
        times=TIMES,
        train_times=TRAIN_TIMES,
        to_cens_shift=[],
        to_term_shift=[],
        cens_prob=-1,
    )

    train_cfg = TrainingConfig(
        epochs=1,
        lr=1e-4,
        early_stopping=True,
        patience=10,
        min_delta=0.001,
        score_metric="ibs",
    )

    exp_cfg = ExperimentConfig(
        metrics=METRICS,
        schema=SCHEMA,
        res_filename=RES_FILE,
        models_folder=MODELS_DIR,
        log_dir=LOG_DIR,
    )

    scorer = ModelScorer()

    BASE_DIR, MODELS_DIR, LOG_DIR, RES_FILE = init_experiments_folder(
        exp_num=EXP_NUM,
        exp_cfg=exp_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        exp_desc=EXP_DESC,
        files_to_save=FILES_TO_SAVE,
    )

    max_train_samples = max(TRAIN_GRID)
    dl_score_max = prepare_dataloader(
        train_samples=max_train_samples,
        data_cfg=data_cfg,
        data_type="train",
        dataset_type="score",
    )

    df_train_max = pd.read_csv(
        f"{DATA_FOLDER}/{max_train_samples}_train_preprocessed.csv"
    )
    df_train_max["duration"] = df_train_max["max_lifetime"] - df_train_max["time"]
    df_train_max = df_train_max[["duration", "failure", "time"]]

    run_id = 0
    for train_samples in TRAIN_GRID:
        for method in METHODS:
            for hparams in PARAM_GRIDS[method]:
                run_single_experiment(
                    train_samples=train_samples,
                    test_grid=TEST_GRID,
                    method=method,
                    hparams=hparams,
                    run_id=run_id,
                    scorer=scorer,
                    dl_score_max=dl_score_max,
                    df_train_max=df_train_max,
                    exp_cfg=exp_cfg,
                    data_cfg=data_cfg,
                    train_cfg=train_cfg,
                    val_data_size=VAL_DATA_SIZE,
                )
                run_id += 1
