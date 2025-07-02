import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from src.data_loader import load_plays

# Utilitário para encontrar o QB
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

# Prepara jogada única
def prepare_single_play_data(play_df: pd.DataFrame, plays_df: pd.DataFrame):
    if play_df['event'].isna().all():
        raise ValueError("Play sem eventos válidos.")

    qb_pos = get_qb_position(play_df)
    if qb_pos is None:
        raise ValueError("Não foi possível inferir a posição do QB.")
    qb_x, qb_y = qb_pos

    play_df = play_df.copy()
    play_df['dist_to_qb'] = np.sqrt((play_df['x'] - qb_x) ** 2 + (play_df['y'] - qb_y) ** 2)

    players = play_df[play_df['nflId'].notna()]
    player_features = players.groupby('nflId')[['x', 'y', 's', 'a', 'o', 'dir', 'dist_to_qb']].mean().values.flatten()

    if len(player_features) != 7 * 22:
        raise ValueError("Jogada incompleta – menos de 22 jogadores com dados válidos.")

    return pd.DataFrame([player_features])

# Dados
tracking = pd.read_csv("data/tracking_week_1.csv")
plays = load_plays()

sample_play = tracking[['gameId', 'playId']].drop_duplicates().sample(1).iloc[0]
gameId = sample_play['gameId']
playId = sample_play['playId']
df = tracking[(tracking['gameId'] == gameId) & (tracking['playId'] == playId)]

# Predição
try:
    X_pred = prepare_single_play_data(df.copy(), plays.copy())
    model = joblib.load("models/trained_model.pkl")
    predicted_yards = model.predict(X_pred)[0]
except Exception as e:
    print(f"Erro ao preparar jogada ou prever jardas: {e}")
    predicted_yards = 0

# Direção do jogo
flip = df['playDirection'].iloc[0] == 'left'
teams = df['club'].dropna().unique()
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors = {team: palette[i % len(palette)] for i, team in enumerate(teams)}

def animate_play(play_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    def draw_field():
        ax.set_facecolor('green')
        for yard in range(10, 110, 10):
            ax.axvline(yard, color='white', linestyle='--', linewidth=0.5)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        draw_field()
        ax.set_title(f"Play {playId} – Predição: {predicted_yards:.2f} jardas", fontsize=14)

        frame_data = play_df[play_df['frameId'] == frame]
        x = 120 - frame_data['x'] if flip else frame_data['x']
        y = 53.3 - frame_data['y'] if flip else frame_data['y']

        # Separa bola dos jogadores
        ball_data = frame_data[frame_data['nflId'].isna()]
        players_data = frame_data[frame_data['nflId'].notna()]

        # Cores dos jogadores
        colors_list = [colors.get(club, 'gray') for club in players_data['club']]
        player_x = 120 - players_data['x'] if flip else players_data['x']
        player_y = 53.3 - players_data['y'] if flip else players_data['y']
        dots = ax.scatter(player_x, player_y, c=colors_list, s=300, edgecolors='black')

        # Bola em preto
        if not ball_data.empty:
            ball_x = 120 - ball_data['x'].values[0] if flip else ball_data['x'].values[0]
            ball_y = 53.3 - ball_data['y'].values[0] if flip else ball_data['y'].values[0]
            ax.scatter(ball_x, ball_y, c='black', s=100, marker='o', edgecolors='white', linewidths=0.5)

        # Números dos jogadores
        for _, row in players_data.iterrows():
            if pd.notna(row['jerseyNumber']):
                x_val = 120 - row['x'] if flip else row['x']
                y_val = 53.3 - row['y'] if flip else row['y']
                ax.text(x_val, y_val, str(int(row['jerseyNumber'])),
                        color='white', ha='center', va='center',
                        fontsize=8, fontweight='bold')

        return [dots]


    frames = sorted(play_df['frameId'].unique())
    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    plt.xlabel("Campo (jardas)")
    plt.ylabel("Largura do campo")
    plt.tight_layout()
    plt.show()

# Executa animação
animate_play(df)
