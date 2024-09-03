import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from typing import Tuple

def split_isic2024(train_metadata_path: str, seed: int, split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
    sr_total = sum(split_ratio)
    sr = [sr_subset/sr_total for sr_subset in split_ratio]
    df = pd.read_csv(train_metadata_path)
    unique_patient_ids = df['patient_id'].unique()
    train_val_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=sr[2])
    train_val_df = df[df['patient_id'].isin(train_val_patient_ids)]
    test_df = df[df['patient_id'].isin(test_patient_ids)]
    train_patient_ids, val_patient_ids = train_test_split(train_val_patient_ids, test_size=sr[1]/sum(sr[:2]))
    train_df = train_val_df[train_val_df['patient_id'].isin(train_patient_ids)]
    val_df = train_val_df[train_val_df['patient_id'].isin(val_patient_ids)]
    return train_df, val_df, test_df

def cv_isic2024(train_metadata_path: str, n_folds: int, seed: int):
    """Splits ISIC_2024 training set, grouping by patient Id. for cross-validation

    Args:
        train_metadata_path (str): path to the csv metadata file.
        n_folds (int): folds numbers.
        seed (int): random seed.

    Returns:
        pandas.DataFrame: train metadata with added column 'fold' (=fold nr with same patient data grouped inside each fold)
    """        
    train_metadata = pd.read_csv(train_metadata_path)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    split = sgkf.split(train_metadata, train_metadata.target, groups=train_metadata.patient_id)
    for i, (_, val_index) in enumerate(split):
        train_metadata.loc[val_index, 'fold'] = i
    return train_metadata

if __name__ == '__main__':
    train_metadata_path = "~/datos/isic_2024_data/train-metadata.csv"
    seed = 42

    train_df, val_df, test_df = split_isic2024(train_metadata_path, seed, split_ratio=(1,0.7,0.3))

    print(f"{train_df}\n\n{val_df}\n\n{test_df}")