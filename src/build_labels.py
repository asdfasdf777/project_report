from collections import Counter
import json

from config import (
    NUM_LABELS,
    SPLITS_DIR,
    MAX_TRACKS
)

from fma_metadata import load_tracks, get_track_tags, get_track_split


def main():

    tracks = load_tracks()

    # Only use tracks from the official training split
    # when determining which tags are most common.
    training_tracks = []

    for track_id in tracks.index:

        split = get_track_split(tracks, track_id)

        if split == "training":
            training_tracks.append(track_id)

    print("Training tracks:", len(training_tracks))

    counter = Counter()

    for track_id in training_tracks:

        tags = get_track_tags(tracks, track_id)

        counter.update(tags)

    most_common = counter.most_common(NUM_LABELS)

    labels = [tag for tag, count in most_common]

    print("\nSelected labels:")

    for i, label in enumerate(labels):
        print(i, label, counter[label])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    output = SPLITS_DIR / "labels.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()