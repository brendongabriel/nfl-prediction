import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_qb_position(group: pd.DataFrame) -> tuple[float, float] | None:
    snap_frame = group[group['event'] == 'ball_snap']
    if snap_frame.empty:
        return None

    ball = snap_frame[snap_frame['nflId'].isna()]
    if ball.empty:
        return None

    ball_x = ball['x'].values[0]
    ball_y = ball['y'].values[0]

    players = snap_frame[snap_frame['nflId'].notna()].copy()
    players['distance'] = ((players['x'] - ball_x)**2 + (players['y'] - ball_y)**2) ** 0.5

    qb = players.loc[players['distance'].idxmin()]
    return qb['x'], qb['y']

def prepare_data(tracking_df: pd.DataFrame, plays_df: pd.DataFrame):
    features = []
    labels = []

    merged = tracking_df.merge(plays_df, on=['gameId', 'playId'], how='left')
    grouped = merged.groupby(['gameId', 'playId'])

    for (gameId, playId), group in grouped:
        if group['event'].isna().all():
            continue

        # Distância ao QB (com inferência de posição)
        qb_pos = get_qb_position(group)
        if qb_pos is None:
            continue
        qb_x, qb_y = qb_pos
        group['dist_to_qb'] = np.sqrt((group['x'] - qb_x)**2 + (group['y'] - qb_y)**2)

        # Média por jogador (exclui bola)
        players = group[group['nflId'].notna()]
        player_features = players.groupby('nflId')[['x', 'y', 's', 'a', 'o', 'dir', 'dist_to_qb']].mean().values.flatten()

        if len(player_features) != 7 * 22:
            continue  # Skip se não houver 22 jogadores válidos

        features.append(player_features)
        labels.append(group['yardsGained'].iloc[0])

    X = pd.DataFrame(features)
    y = pd.Series(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
