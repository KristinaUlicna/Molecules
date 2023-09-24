import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

URL_DATA = "https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv"
CUT_OFF_THRESHOLD = 8.0


def read_data_from_url(
    url_file_path: str = URL_DATA,
) -> pd.DataFrame:
    df = pd.read_csv(url_file_path)
    return df


def extract_relevant_data(
    data_frame: pd.DataFrame
) -> tuple[list[str], list[str], list[float]]:
    names_data = data_frame["molecule_chembl_id"].tolist()
    smile_data = data_frame["smiles"].tolist()
    pIC50_data = data_frame["pIC50"].tolist()
    assert len(names_data) == len(smile_data) == len(pIC50_data)
    return names_data, smile_data, pIC50_data


def shuffle_lists_together(
    names: list[str], 
    smile: list[str], 
    pIC50: list[float],
) -> tuple[list[str], list[str], list[float]]:
    # Define random seed for reproducibility
    random.seed(23)

    # Combine, shuffle & unpack:
    combined_lists = list(zip(names, smile, pIC50))
    random.shuffle(combined_lists)
    names, smile, pIC50 = zip(*combined_lists)

    # Convert the tuples back to lists & return
    return list(names), list(smile), list(pIC50)


def get_splitter_indices(
    num_data_points: int,
    train_valid_infer_split: tuple[float], 
) -> tuple[tuple[int, int]]:

    train_splitter = 0, round(train_valid_infer_split[0] * num_data_points)
    valid_splitter = train_splitter[1], round((train_valid_infer_split[0] + train_valid_infer_split[1]) * num_data_points)
    infer_splitter = valid_splitter[1], num_data_points
    return train_splitter, valid_splitter, infer_splitter


def split_graphs_dataset(
    total_smiles_data: list[str],
    total_pIC50_value: list[float],
    train_valid_infer_split: tuple[float] = (0.8, 0.1, 0.1), 
):
    num_data_points = len(total_smiles_data)
    splitters = get_splitter_indices(num_data_points, train_valid_infer_split)
    train_splitter, valid_splitter, infer_splitter =  splitters

    train_smile_data = total_smiles_data[train_splitter[0]:train_splitter[1]]
    train_pIC50_data = total_pIC50_value[train_splitter[0]:train_splitter[1]]
    assert len(train_smile_data) == len(train_pIC50_data)

    valid_smile_data = total_smiles_data[valid_splitter[0]:valid_splitter[1]]
    valid_pIC50_data = total_pIC50_value[valid_splitter[0]:valid_splitter[1]]
    assert len(valid_smile_data) == len(valid_pIC50_data)

    infer_smile_data = total_smiles_data[infer_splitter[0]:infer_splitter[1]]
    infer_pIC50_data = total_pIC50_value[infer_splitter[0]:infer_splitter[1]]
    assert len(infer_smile_data) == len(infer_pIC50_data)

    return {
        "train" : [train_smile_data, train_pIC50_data], 
        "valid" : [valid_smile_data, valid_pIC50_data],
        "infer" : [infer_smile_data, infer_pIC50_data],
    }
