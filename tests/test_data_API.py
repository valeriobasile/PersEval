import pytest
from unittest.mock import Mock
import perseval.data as data_module

from perseval.data import (available_datasets, download, download_and_split)

def test_available_datasets():
    datasets = available_datasets()
    
    assert isinstance (datasets, list)
    
    assert "epic" in datasets
    assert "brexit" in datasets
    assert "dices" in datasets
    assert "mhs" in datasets
    assert "md" in datasets

def test_download_invalid_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download("does_not_exist")
    
def test_download_requires_string():
    with pytest.raises(TypeError, match="dataset_name must be a string"):
        download(123)
        
def test_download_is_case_insensitive(monkeypatch):
    fake_dataset = Mock()

    monkeypatch.setitem(
        data_module._DATASETS,
        "epic",
        lambda: fake_dataset,
    )

    assert download("epic") is fake_dataset
    assert download("EPIC") is fake_dataset
    assert download("Epic") is fake_dataset
    
def test_download_epic(monkeypatch):
    fake_dataset = Mock()

    factory = Mock(return_value=fake_dataset)

    monkeypatch.setitem(
        data_module._DATASETS,
        "epic",
        factory,
    )

    result = download("epic")

    factory.assert_called_once_with()
    assert result is fake_dataset
    
def test_available_labels():
    dataset = data_module.PerspectivistDataset()

    dataset.labels = {
        "hs": set(),
        "offensiveness": set(),
        "aggressiveness": set(),
        "stereotype": set(),
    }

    assert dataset.available_labels() == [
        "hs",
        "offensiveness",
        "aggressiveness",
        "stereotype",
    ]
    
def test_download_and_split(monkeypatch):
    fake_dataset = Mock()

    download_mock = Mock(return_value=fake_dataset)

    monkeypatch.setattr(
        data_module,
        "download",
        download_mock,
    )

    result = download_and_split(
        "epic",
        user_adaptation="train",
        extended=True,
        named=True,
        baseline=False,
    )

    download_mock.assert_called_once_with("epic")

    fake_dataset.get_splits.assert_called_once_with(
        extended=True,
        user_adaptation="train",
        named=True,
        baseline=False,
    )

    assert result is fake_dataset
    
def test_public_api():
    import perseval

    assert callable(perseval.download)
    assert callable(perseval.download_and_split)
    assert callable(perseval.available_datasets)