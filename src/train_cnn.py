import ast
import json
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.metrics import (
    f1_score,
    average_precision_score
)

from config import (
    FMA_AUDIO_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    NUM_LABELS,
    SAMPLE_RATE,
    EPOCHS
)

from cnn_model import MusicCNN


BATCH_SIZE = 8
LEARNING_RATE = 0.001

MEL_BINS = 128

AUDIO_SECONDS = 10

HOP_LENGTH = 512


def set_seed(
    seed=42
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def find_audio_file(
    track_id
):

    track_id = int(
        track_id
    )

    directory = (
        FMA_AUDIO_DIR
        / f"{track_id // 1000:03d}"
    )

    filename = (
        f"{track_id:06d}.mp3"
    )

    return (
        directory
        / filename
    )


def load_mel_spectrogram(
    audio_path
):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=AUDIO_SECONDS
    )

    target_length = (
        SAMPLE_RATE
        * AUDIO_SECONDS
    )

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (
                0,
                target_length
                - len(audio)
            )
        )

    else:

        audio = audio[
            :target_length
        ]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=MEL_BINS,
        hop_length=HOP_LENGTH,
        n_fft=2048
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each spectrogram.
    mel_db = (
        mel_db
        - mel_db.mean()
    )

    std = (
        mel_db.std()
        + 1e-8
    )

    mel_db = (
        mel_db
        / std
    )

    return mel_db.astype(
        np.float32
    )


class MusicCNNDataset(
    Dataset
):

    def __init__(
        self,
        dataframe
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe.iloc[
                index
            ]
        )

        track_id = int(
            row["track_id"]
        )

        audio_path = find_audio_file(
            track_id
        )

        mel = load_mel_spectrogram(
            audio_path
        )

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # Add channel dimension.
        mel = mel.unsqueeze(
            0
        )

        labels = torch.tensor(
            ast.literal_eval(
                row["labels"]
            ),
            dtype=torch.float32
        )

        return {
            "mel": mel,
            "labels": labels,
            "track_id": track_id
        }


def calculate_metrics(
    probabilities,
    labels
):

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.float32
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_auc = average_precision_score(
        labels,
        probabilities,
        average="macro"
    )

    micro_auc = average_precision_score(
        labels,
        probabilities,
        average="micro"
    )

    return {
        "macro_f1": float(
            macro_f1
        ),
        "micro_f1": float(
            micro_f1
        ),
        "macro_auc_pr": float(
            macro_auc
        ),
        "micro_auc_pr": float(
            micro_auc
        )
    }


def evaluate(
    model,
    loader,
    device
):

    model.eval()

    all_probabilities = []

    all_labels = []

    all_track_ids = []

    with torch.no_grad():

        for batch in loader:

            mel = (
                batch["mel"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            logits = model(
                mel
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_track_ids.extend(
                batch[
                    "track_id"
                ].tolist()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    metrics = calculate_metrics(
        probabilities,
        labels
    )

    return (
        metrics,
        probabilities,
        labels,
        all_track_ids
    )


def main():

    print(
        "Starting CNN mel-spectrogram baseline..."
    )

    set_seed()

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "Device:",
        device
    )

    dataset_path = (
        SPLITS_DIR
        / "context_dataset.csv"
    )

    dataframe = pd.read_csv(
        dataset_path
    )

    train_df = dataframe[
        dataframe["split"] == "training"
    ].copy()

    val_df = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------

    train_dataset = MusicCNNDataset(
        train_df
    )

    val_dataset = MusicCNNDataset(
        val_df
    )

    test_dataset = MusicCNNDataset(
        test_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print(
        "\nCreating CNN..."
    )

    model = MusicCNN(
        NUM_LABELS
    )

    model = model.to(
        device
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
        "val_macro_auc_pr": [],
        "val_micro_auc_pr": []
    }

    print(
        "\nBeginning training..."
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for batch_index, batch in enumerate(
            train_loader
        ):

            mel = (
                batch["mel"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                mel
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            if (
                batch_index + 1
            ) % 25 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_index + 1}/"
                    f"{len(train_loader)} "
                    f"Loss={loss.item():.4f}"
                )

        average_loss = (
            total_loss
            /
            len(train_loader)
        )

        (
            val_metrics,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            device
        )

        history[
            "train_loss"
        ].append(
            average_loss
        )

        history[
            "val_macro_f1"
        ].append(
            val_metrics[
                "macro_f1"
            ]
        )

        history[
            "val_micro_f1"
        ].append(
            val_metrics[
                "micro_f1"
            ]
        )

        history[
            "val_macro_auc_pr"
        ].append(
            val_metrics[
                "macro_auc_pr"
            ]
        )

        history[
            "val_micro_auc_pr"
        ].append(
            val_metrics[
                "micro_auc_pr"
            ]
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Average Loss={average_loss:.4f}"
        )

        print(
            f"Validation Macro-F1="
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Validation Micro-F1="
            f"{val_metrics['micro_f1']:.4f}"
        )

        print(
            f"Validation Macro AUC-PR="
            f"{val_metrics['macro_auc_pr']:.4f}"
        )

        print(
            f"Validation Micro AUC-PR="
            f"{val_metrics['micro_auc_pr']:.4f}"
        )

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

    print(
        "\nEvaluating on test set..."
    )

    (
        test_metrics,
        probabilities,
        labels,
        track_ids
    ) = evaluate(
        model,
        test_loader,
        device
    )

    print(
        "\nFINAL CNN TEST RESULTS"
    )

    print(
        f"Macro-F1: "
        f"{test_metrics['macro_f1']}"
    )

    print(
        f"Micro-F1: "
        f"{test_metrics['micro_f1']}"
    )

    print(
        f"Macro AUC-PR: "
        f"{test_metrics['macro_auc_pr']}"
    )

    print(
        f"Micro AUC-PR: "
        f"{test_metrics['micro_auc_pr']}"
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_dir = (
        RESULTS_DIR
        / "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS_DIR
        / "cnn_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    prediction_path = (
        predictions_dir
        / "cnn_predictions.npz"
    )

    np.savez(
        prediction_path,
        probabilities=probabilities,
        labels=labels,
        track_ids=np.asarray(
            track_ids
        )
    )

    results = {
        "model": "CNN_mel_spectrogram",
        "metrics": test_metrics,
        "history": history
    }

    results_path = (
        RESULTS_DIR
        / "cnn_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nCNN model saved to:",
        model_path
    )

    print(
        "CNN predictions saved to:",
        prediction_path
    )

    print(
        "CNN results saved to:",
        results_path
    )


if __name__ == "__main__":
    main()