import operator
import random
from collections import defaultdict, Counter
from itertools import count
from pathlib import Path
from random import sample
from typing import Literal

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from project2_speechcommands.config import AudioConfig, BalanceConfig


CORE_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = 10
UNKNOWN_LABEL = 11
NUM_CLASSES = 12

PRELIM_KNOWN_LABEL   = 0  # core commands (raw 0-9) → 0
PRELIM_UNKNOWN_LABEL = 1  # UNKNOWN_LABEL (11)      → 1
PRELIM_SILENCE_LABEL = 2  # SILENCE_LABEL (10)      → 2

PRELIM_CLASS_NAMES = ["known", "unknown", "silence"]


def remap_label_for_prelim(raw_label: int) -> int:
    if raw_label == SILENCE_LABEL:
        return PRELIM_SILENCE_LABEL
    if raw_label == UNKNOWN_LABEL:
        return PRELIM_UNKNOWN_LABEL
    return PRELIM_KNOWN_LABEL

# Approximate log-Mel statistics for Speech Commands v1 — update after computing on training set
SC_MEAN = -21.5
SC_STD = 16.0


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: Literal["train", "val", "test"],
        audio_cfg: AudioConfig,
        remap_prelim: bool = False,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.audio_cfg = audio_cfg
        self.remap_prelim = remap_prelim

        # 1. Call _parse_split_files() to get val_set and test_set
        self.val_set, self.test_set = self._parse_split_files()

        # 2. Call _collect_samples() to build self.samples: list[tuple[Path, int, int | None]]
        self.samples = self._collect_samples()
        n = len([x for x in self.samples if x[1] != UNKNOWN_LABEL]) // len(CORE_COMMANDS)

        # 3. Extend with silence samples from _generate_silence_samples()
        self.samples.extend(self._generate_silence_samples(n))

        # 4. Build torchaudio MelSpectrogram transform with audio_cfg params
        self.mel_transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_cfg.sample_rate,
            n_mels=self.audio_cfg.n_mels,
            n_fft=self.audio_cfg.n_fft,
            hop_length=self.audio_cfg.hop_length,
            f_min=self.audio_cfg.f_min,
            f_max=self.audio_cfg.f_max,
        )

    def _parse_split_files(self) -> tuple[set[str], set[str]]:
        """Read validation_list.txt and testing_list.txt.
        Returns (val_set, test_set) where each entry is 'word/filename.wav'."""
        def list_file_to_set(file: Path) -> set[str]:
            with file.open("r") as f:
                return set(f.read().splitlines())
        return (list_file_to_set(self.data_root / "train" / "validation_list.txt"),
                list_file_to_set(self.data_root / "train" / "testing_list.txt"))

    def _collect_samples(self) -> list[tuple[Path, int, int | None]]:
        """Walk data_root/train/, assign labels:
        - word in CORE_COMMANDS → CORE_COMMANDS.index(word)
        - _background_noise_ → skip (handled separately)
        - everything else → UNKNOWN_LABEL
        Filter by split membership using parsed split files."""
        samples = []
        for word_dir in (self.data_root / "train" / "audio").iterdir():
            if word_dir.name == "_background_noise_": continue
            label = CORE_COMMANDS.index(word_dir.name) if word_dir.name in CORE_COMMANDS else UNKNOWN_LABEL
            for wav in word_dir.glob("*.wav"):
                key = f"{word_dir.name}/{wav.name}"
                # assign to train/val/test based on key membership
                if key in self.val_set:
                    file_split = "val"
                elif key in self.test_set:
                    file_split = "test"
                else:
                    file_split = "train"

                if file_split == self.split:
                    samples.append((wav, label, None))
        return samples

    def _generate_silence_samples(self, n: int) -> list[tuple[Path, int, int | None]]:
        """Sample n random 1-second windows from _background_noise_/ WAV files.
        Returns list of (noise_file_path, start_sample_offset, SILENCE_LABEL).
        n should be approximately the size of the smallest core command class."""
        noise_files = list((self.data_root / "train" / "audio" / "_background_noise_").glob("*.wav"))
        noise_samples = []
        num_samples_cache: dict[Path, int] = {}
        for i in range(n):
            noise_file_path = random.choice(noise_files)
            if noise_file_path not in num_samples_cache:
                num_samples_cache[noise_file_path] = sf.info(noise_file_path).frames
            max_offset = num_samples_cache[noise_file_path]
            start_sample_offset = random.randint(0, max_offset)
            noise_samples.append((noise_file_path, SILENCE_LABEL, start_sample_offset))
        return noise_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Pipeline:
        1. Load WAV with sf.read()
        2. Resample to audio_cfg.sample_rate if needed
        3. Pad or trim to audio_cfg.target_length samples
        4. Compute log-Mel: MelSpectrogram → log(spec + 1e-9)  shape: (1, n_mels, time_frames)
        5. Normalize: (spec - SC_MEAN) / SC_STD
        6. Pad time axis to audio_cfg.target_frames
        Returns: (tensor of shape (1, 128, 112), label)
        """
        assert 0 <= idx < len(self)
        # Step 1
        selected_sample = self.samples[idx]
        label = selected_sample[1]
        # Check if it's a silence sample
        if selected_sample[2] is not None:
            data, sr = sf.read(selected_sample[0], start=selected_sample[2], stop=selected_sample[2] + self.audio_cfg.target_length, dtype='float32', always_2d=True)
        else:
            data, sr = sf.read(selected_sample[0], dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channels, samples)

        # Step 2
        if sr != self.audio_cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.audio_cfg.sample_rate)

        # Step 3
        # Check if it needs padding, it cant be longer than target frames.
        if waveform.shape[1] != self.audio_cfg.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.audio_cfg.target_length - waveform.shape[-1]))

        # Step 4
        spec = self.mel_transformer(waveform)
        logMel = torch.log(spec + 1e-9)

        # Step 5
        normalised_logMel = (logMel - SC_MEAN) / SC_STD

        # Step 6
        if normalised_logMel.shape[1] != self.audio_cfg.target_frames:
            normalised_logMel = torch.nn.functional.pad(normalised_logMel, (0, self.audio_cfg.target_frames - normalised_logMel.shape[-1]))

        if self.remap_prelim:
            label = remap_label_for_prelim(label)
        return normalised_logMel, label


def get_dataloaders(
    root: str | Path,
    audio_cfg: AudioConfig,
    balance_cfg: BalanceConfig,
    batch_size: int = 64,
    num_workers: int = 4,
    test_mode: bool = False,
    remap_prelim: bool = False,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    If test_mode=True, returns (None, None, test_loader).

    Balance strategies:
    - "none": standard shuffle DataLoader
    - "oversample": WeightedRandomSampler with inverse-frequency weights
    - "prelim": standard DataLoader with 3-class label remapping (known/unknown/silence)
    """
    test_dataset = SpeechCommandsDataset(root, "test", audio_cfg=audio_cfg, remap_prelim=remap_prelim)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    if test_mode:
        return None, None, test_dataloader

    val_dataset = SpeechCommandsDataset(root, "val", audio_cfg=audio_cfg, remap_prelim=remap_prelim)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    train_dataset = SpeechCommandsDataset(root, "train", audio_cfg=audio_cfg, remap_prelim=remap_prelim)

    if balance_cfg.strategy == "oversample":
        class_counts = Counter(label for _, label, _ in train_dataset.samples)
        weights = [1.0 / class_counts[label] for _, label, _ in train_dataset.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
