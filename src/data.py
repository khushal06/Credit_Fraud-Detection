import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE
except ModuleNotFoundError:  # pragma: no cover
    from config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Credit card fraud dataset is missing. Download it from Kaggle: "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and save it as "
            f"{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}.")
    return df


def split_dataset():
    df = load_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    scale_columns = ["Time", "Amount"]
    for column in scale_columns:
        if column not in X_train.columns:
            raise ValueError(f"Expected required column '{column}' in the dataset.")

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])

    return X_train, X_test, y_train, y_test, scaler
