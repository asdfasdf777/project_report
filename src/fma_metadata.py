import ast
import pandas as pd

from config import FMA_METADATA_DIR


def load_tracks():
    path = FMA_METADATA_DIR / "tracks.csv"

    tracks = pd.read_csv(
        path,
        index_col=0,
        header=[0, 1]
    )

    return tracks


def get_track_tags(tracks, track_id):
    value = tracks.loc[track_id, ("track", "tags")]

    if pd.isna(value):
        return []

    try:
        tags = ast.literal_eval(value)

        if isinstance(tags, list):
            return [str(tag).lower().strip() for tag in tags]

    except Exception:
        pass

    return []


def get_track_split(tracks, track_id):
    return tracks.loc[track_id, ("set", "split")]


def get_track_genre(tracks, track_id):
    return tracks.loc[track_id, ("track", "genre_top")]


if __name__ == "__main__":
    tracks = load_tracks()

    print("Number of tracks:", len(tracks))

    first_track = tracks.index[0]

    print("Example track:", first_track)
    print("Tags:", get_track_tags(tracks, first_track))
    print("Split:", get_track_split(tracks, first_track))
    print("Genre:", get_track_genre(tracks, first_track))