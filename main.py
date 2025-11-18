import os
import uuid
import math
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

DATA_DIR = "../data/"
SONGS_FILE = "songs.csv"
NFEATURE = 21
S = 50
totReco = 0
startConstant = 5

songs_path = os.path.join(DATA_DIR, SONGS_FILE)
if not os.path.exists(songs_path):
    raise FileNotFoundError(f"Songs file not found at: {songs_path}")

Songs = pd.read_csv(songs_path, index_col=0)

if 'last_t' not in Songs.columns:
    Songs['last_t'] = 0

users = {}

def register_user():
    name = input("Enter your name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return None
    for uid, info in users.items():
        if info.get("name") == name:
            print("A user with this name already exists. Please login or choose a different name.")
            return None

    user_id = str(uuid.uuid4())
    users[user_id] = {
        "name": name,
        "features": np.zeros(NFEATURE, dtype=float),
        "rated_songs": set()
    }
    print(f"User {name} registered successfully. Your user ID is {user_id}")
    return user_id

def login_user():
    user_id = input("Enter your user ID: ").strip()
    if user_id not in users:
        print("Invalid user ID. Please register first.")
        return None
    print(f"Welcome back, {users[user_id]['name']}!")
    return user_id

def get_user_data(user_id):
    return users.get(user_id)

def compute_utility(user_features, song_features, epoch, s=S):
    user_features = np.asarray(user_features, dtype=float).copy()
    song_features = np.asarray(song_features, dtype=float).copy()
    dot = float(user_features.dot(song_features))
    ee = (1.0 - 1.0 * math.exp(-1.0 * float(epoch) / float(s)))
    res = dot * ee
    return res

def get_song_features(song):
    if isinstance(song, pd.DataFrame):
        if song.shape[0] == 1:
            song = song.iloc[0]
        else:
            raise TypeError("DataFrame must have exactly one row to extract features.")
    if isinstance(song, pd.Series):
        return np.asarray(song.values[-NFEATURE:], dtype=float)
    else:
        raise TypeError(f"{type(song)} provided; expected pandas Series or single-row DataFrame")

def get_song_genre(song):
    genres = []
    possible_genres = [
        "Pop", "Rock", "Country", "Folk", "Dance", "Grunge", "Love", "Metal",
        "Classic", "Funk", "Electric", "Acoustic", "Indie", "Jazz", "SoundTrack", "Rap"
    ]
    for genre in possible_genres:
        val = song.get(genre, 0) if isinstance(song, pd.Series) else 0
        try:
            if int(val) == 1:
                genres.append(genre)
        except Exception:
            continue
    return genres

def best_recommendation(user_features, epoch, s):
    utilities = np.zeros(Songs.shape[0], dtype=float)
    for i, (idx, song) in enumerate(Songs.iterrows()):
        last_t = int(song.get('last_t', 0))
        song_features = get_song_features(song)
        utilities[i] = compute_utility(user_features, song_features, epoch - last_t, s)
    best_idx = utilities.argmax()
    return Songs.iloc[[best_idx]]

def random_choice(exclude_set=None):
    if exclude_set is None:
        exclude_set = set()
    candidates = Songs.loc[~Songs.index.isin(list(exclude_set))]
    if candidates.empty:
        return Songs.sample(n=1)
    return candidates.sample(n=1)

def greedy_choice(user_features, epoch, s):
    global totReco
    epsilon = 1.0 / math.sqrt(max(1, epoch + 1))
    totReco += 1
    if random.random() > epsilon:
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def greedy_choice_no_t(user_features, epoch, s, epsilon=0.3):
    global totReco
    totReco += 1
    if random.random() > epsilon:
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def all_recommendation(user_features, user_rated_songs_set):
    recoSongs = []
    attempts = 0
    while len(recoSongs) < 10 and attempts < 50:
        song = greedy_choice_no_t(user_features, totReco, S)
        idx = song.index[0]
        if idx in user_rated_songs_set:
            attempts += 1
            continue
        Songs.loc[Songs.index == idx, 'last_t'] = totReco
        recoSongs.append(song)
        attempts += 1
    return recoSongs

def iterative_mean(old, new, t):
    t = float(t) + startConstant
    return ((t - 1.0) / t) * np.asarray(old, dtype=float) + (1.0 / t) * np.asarray(new, dtype=float)

def update_features(user_features, song_features, rating, t):
    sample = song_features * float(rating)
    return iterative_mean(user_features, sample, t)

def reinforcement_learning(user_id, s=200, N=5):
    user_data = get_user_data(user_id)
    if user_data is None:
        print("User not found.")
        return

    user_features = user_data["features"]
    rated_songs = user_data["rated_songs"]

    Features = [
        "1980s", "1990s", "2000s", "2010s", "2020s",
        "Pop", "Rock", "Country", "Folk", "Dance", "Grunge",
        "Love", "Metal", "Classic", "Funk", "Electric", "Acoustic",
        "Indie", "Jazz", "SoundTrack", "Rap"
    ]

    print("Select song features that you like (enter numbers).")
    for i, feat in enumerate(Features, start=1):
        print(f"{i}. {feat}")

    likedFeat = set()
    while True:
        num = input("Enter number associated with feature (or blank to finish): ").strip()
        if num == "":
            break
        if not num.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(num) - 1
        if idx < 0 or idx >= len(Features):
            print("Number out of range.")
            continue
        likedFeat.add(Features[idx])
        more = input("Add another feature? (y/n): ").strip().lower()
        if more != 'y':
            break

    if likedFeat:
        weight = 1.0 / len(likedFeat)
        for i, feat in enumerate(Features):
            if feat in likedFeat:
                user_features[i] = weight

    print(f"\n\nRate the following {N} songs (1-10 scale).\n")
    for t in range(N):
        if t >= 10:
            recommendation = greedy_choice_no_t(user_features, t + 1, s, epsilon=0.3)
        else:
            recommendation = greedy_choice(user_features, t + 1, s)

        attempts = 0
        while recommendation.index[0] in rated_songs and attempts < 10:
            recommendation = random_choice(exclude_set=rated_songs)
            attempts += 1

        recommendation_features = get_song_features(recommendation)

        while True:
            user_rating = input(f'How much do you like "{recommendation.index[0]}" (1-10): ').strip()
            if not user_rating.isdigit():
                print("Please enter an integer rating between 1 and 10.")
                continue
            user_rating = int(user_rating)
            if not (1 <= user_rating <= 10):
                print("Rating must be between 1 and 10.")
                continue
            break

        user_rating = float(user_rating) / 10.0

        user_features = update_features(user_features, recommendation_features, user_rating, t + 1)
        Songs.loc[Songs.index.isin([recommendation.index[0]]), 'last_t'] = t + 1
        rated_songs.add(recommendation.index[0])

        user_data["features"] = user_features
        user_data["rated_songs"] = rated_songs

    print("\n\nBased on your preferences, here are some recommendations:\n")
    recos = all_recommendation(user_features, rated_songs)
    for i, song in enumerate(recos):
        print(f"{i+1}. {song.index[0]}  Genres: {', '.join(get_song_genre(song.iloc[0]))}")

def main():
    choice = input("Do you want to (r)egister or (l)ogin? (r/l): ").strip().lower()
    user_id = None
    if choice == 'r':
        user_id = register_user()
    elif choice == 'l':
        user_id = login_user()
    else:
        print("Invalid choice. Exiting.")
        return

    if user_id:
        reinforcement_learning(user_id)

if __name__ == "__main__":
    main()
