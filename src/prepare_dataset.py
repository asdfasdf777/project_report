import ast
import json
import pandas as pd
import numpy as np

from config import (
    FMA_METADATA_DIR,
    SPLITS_DIR,
    PROCESSED_DIR
)


TRAIN_SAMPLES = 800
VALIDATION_SAMPLES = 100
TEST_SAMPLES = 100


def load_tracks():

    path = FMA_METADATA_DIR / "tracks.csv"

    return pd.read_csv(
        path,
        index_col=0,
        header=[0, 1]
    )


def parse_tags(value):

    if pd.isna(value):

        return []

    try:

        tags = ast.literal_eval(value)

        if isinstance(tags, list):

            return [
                str(tag).lower().strip()
                for tag in tags
            ]

    except Exception:

        pass

    return []


def graph_exists(track_id):

    graph_path = (
        PROCESSED_DIR /
        "graphs" /
        f"{int(track_id):06d}.pt"
    )

    return graph_path.exists()


def collect_tracks(
    tracks,
    split,
    target_count,
    labels,
    label_to_index
):

    rows = []

    for track_id in tracks.index:

        current_split = tracks.loc[
            track_id,
            ("set", "split")
        ]

        if current_split != split:

            continue

        if not graph_exists(track_id):

            continue

        tags = parse_tags(
            tracks.loc[
                track_id,
                ("track", "tags")
            ]
        )

        usable_tags = [
            tag
            for tag in tags
            if tag in label_to_index
        ]

        if len(usable_tags) == 0:

            continue

        vector = np.zeros(
            len(labels),
            dtype=np.float32
        )

        for tag in usable_tags:

            vector[
                label_to_index[tag]
            ] = 1.0

        genre = tracks.loc[
            track_id,
            ("track", "genre_top")
        ]

        rows.append({
            "track_id": int(track_id),
            "split": split,
            "text": ", ".join(usable_tags),
            "genre": str(genre),
            "labels": vector.tolist()
        })

        if len(rows) >= target_count:

            break

    return rows


def main():

    print("Loading FMA metadata...")

    tracks = load_tracks()

    print(
        "Total metadata tracks:",
        len(tracks)
    )

    labels_path = (
        SPLITS_DIR / "labels.json"
    )

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as f:

        labels = json.load(f)

    print(
        "Number of labels:",
        len(labels)
    )

    label_to_index = {
        label: i
        for i, label in enumerate(labels)
    }

    print("\nCollecting training tracks...")

    training_rows = collect_tracks(
        tracks,
        "training",
        TRAIN_SAMPLES,
        labels,
        label_to_index
    )

    print(
        "Training samples:",
        len(training_rows)
    )

    print("\nCollecting validation tracks...")

    validation_rows = collect_tracks(
        tracks,
        "validation",
        VALIDATION_SAMPLES,
        labels,
        label_to_index
    )

    print(
        "Validation samples:",
        len(validation_rows)
    )

    print("\nCollecting test tracks...")

    test_rows = collect_tracks(
        tracks,
        "test",
        TEST_SAMPLES,
        labels,
        label_to_index
    )

    print(
        "Test samples:",
        len(test_rows)
    )

    rows = (
        training_rows +
        validation_rows +
        test_rows
    )

    dataframe = pd.DataFrame(rows)

    output = (
        SPLITS_DIR / "dataset.csv"
    )

    dataframe.to_csv(
        output,
        index=False
    )

    print(
        "\nDataset saved to:",
        output
    )

    print(
        "Total samples:",
        len(dataframe)
    )

    print(
        "\nFinal split counts:"
    )

    print(
        dataframe["split"].value_counts()
    )


if __name__ == "__main__":

    main()