import ast
import json
from pathlib import Path

import pandas as pd
import torch

from config import (
    FMA_AUDIO_DIR,
    FMA_METADATA_DIR,
    PROCESSED_DIR
)

from audio_features import (
    load_audio,
    extract_segment_features
)

from graph_builder import build_graph


def find_audio_file(track_id):

    track_id = int(track_id)

    directory = (
        FMA_AUDIO_DIR /
        f"{track_id // 1000:03d}"
    )

    filename = f"{track_id:06d}.mp3"

    path = directory / filename

    return path


def main():

    metadata_path = (
        FMA_METADATA_DIR /
        "tracks.csv"
    )

    tracks = pd.read_csv(
        metadata_path,
        index_col=0,
        header=[0, 1]
    )

    output_directory = (
        PROCESSED_DIR /
        "graphs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0

    for track_id in tracks.index:

        split = tracks.loc[
            track_id,
            ("set", "split")
        ]

        if split not in [
            "training",
            "validation",
            "test"
        ]:
            continue

        audio_path = find_audio_file(
            track_id
        )

        if not audio_path.exists():

            print(
                "Missing:",
                audio_path
            )

            continue

        try:

            audio, sr = load_audio(
                audio_path
            )

            features = extract_segment_features(
                audio,
                sr
            )

            if len(features) == 0:
                continue

            graph = build_graph(
                features
            )

            graph.track_id = int(track_id)

            graph.split = split

            output_path = (
                output_directory /
                f"{int(track_id):06d}.pt"
            )

            torch.save(
                graph,
                output_path
            )

            processed += 1

            print(
                f"Processed {processed}: "
                f"{track_id}"
            )

        except Exception as error:

            print(
                "Error processing",
                track_id,
                error
            )

    print(
        "\nGraphs created:",
        processed
    )


if __name__ == "__main__":
    main()