# src/data_loader.py

import pandas as pd
import os

DATA_DIR = "data" 

def load_tracking_data(week: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"tracking_week_{week}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_players() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "players.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_games() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "games.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_plays() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "plays.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)
