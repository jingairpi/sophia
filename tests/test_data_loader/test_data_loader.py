import logging
import os
import tempfile

import numpy as np
import pytest

from sophia.data.data_loader import DataLoader, load_binary_data

logger = logging.getLogger(__name__)


def create_temp_binary_file(data: np.ndarray) -> str:
    temp_fd, temp_path = tempfile.mkstemp(suffix=".bin")
    os.close(temp_fd)
    data.tofile(temp_path)
    return temp_path


def test_load_binary_data_success():
    original_data = np.arange(100, dtype=np.int32)
    temp_path = create_temp_binary_file(original_data)
    try:
        loaded_data = load_binary_data(temp_path, memmap=False, dtype=np.int32)
        np.testing.assert_array_equal(loaded_data, original_data)
    finally:
        os.remove(temp_path)


def test_load_binary_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_binary_data("non_existent_file.bin")


def test_dataloader_batches():
    original_data = np.arange(120, dtype=np.int32)
    temp_path = create_temp_binary_file(original_data)
    try:
        loader = DataLoader(
            file_path=temp_path, batch_size=2, seq_len=10, repeat=False, shuffle=False
        )
        batches = list(iter(loader))
        assert len(batches) == 6, f"Expected 6 batches, got {len(batches)}"
        for inputs, targets in batches:
            assert inputs.shape == (2, 9)
            assert targets.shape == (2, 9)
            np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
    finally:
        os.remove(temp_path)


def test_dataloader_reset_and_repeat():
    original_data = np.arange(80, dtype=np.int32)
    temp_path = create_temp_binary_file(original_data)
    try:
        loader = DataLoader(
            file_path=temp_path, batch_size=2, seq_len=10, repeat=True, shuffle=False
        )
        iter_loader = iter(loader)
        all_batches = []
        for _ in range(6):
            batch = next(iter_loader)
            all_batches.append(batch)
        np.testing.assert_array_equal(all_batches[0][0], all_batches[4][0])
        np.testing.assert_array_equal(all_batches[0][1], all_batches[4][1])
    finally:
        os.remove(temp_path)


def test_dataloader_shuffling():
    original_data = np.arange(100, dtype=np.int32)
    temp_path = create_temp_binary_file(original_data)
    try:
        loader1 = DataLoader(
            file_path=temp_path, batch_size=2, seq_len=10, repeat=False, shuffle=True
        )
        loader2 = DataLoader(
            file_path=temp_path, batch_size=2, seq_len=10, repeat=False, shuffle=True
        )

        data1 = np.concatenate([batch[0].flatten() for batch in loader1])
        data2 = np.concatenate([batch[0].flatten() for batch in loader2])

        assert not np.array_equal(
            data1, data2
        ), "Shuffling did not change the data order (or order is identical by chance)."
    finally:
        os.remove(temp_path)
