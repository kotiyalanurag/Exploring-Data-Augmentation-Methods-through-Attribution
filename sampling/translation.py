import pandas as pd

def get_stratified_subsets(df, importance_col="importance", label_col="label", frac=0.1, random_state=42):
    """
    Selects top-10%, bottom-10%, and random-10% samples from each class based on the importance scores.

    Parameters:
    - df: DataFrame containing the dataset.
    - importance_col: Column containing importance scores.
    - label_col: Column containing class labels.
    - frac: Fraction of samples to pick from each class.
    - random_state: Random seed for reproducibility.

    Returns:
    - top_subset: DataFrame with top-10% samples.
    - bottom_subset: DataFrame with bottom-10% samples.
    - random_subset: DataFrame with random-10% samples.
    """
    top_subset = df.apply(lambda x: x.nlargest(int(len(x) * frac), importance_col))
    bottom_subset = df.apply(lambda x: x.nsmallest(int(len(x) * frac), importance_col))
    random_subset = df.apply(lambda x: x.sample(frac=frac, random_state=random_state))

    return top_subset, bottom_subset, random_subset