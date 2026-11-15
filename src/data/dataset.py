import pandas as pd
from pathlib import Path
def load_dataset(path):
    df= pd.read_csv(path)
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    return df