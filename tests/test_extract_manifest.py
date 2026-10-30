"""Tests for scripts/01_extract_manifest.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load scripts/01_extract_manifest.py directly
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_manifest_spec = importlib.util.spec_from_file_location(
    "scripts_01_extract_manifest", _SCRIPTS_DIR / "01_extract_manifest.py"
)
manifest = importlib.util.module_from_spec(_manifest_spec)
_manifest_spec.loader.exec_module(manifest)


class MockClip:
    """Mock DaVinci Resolve clip object."""

    def __init__(self, properties: dict[str, str], metadata: dict[str, str] | None = None, name: str = "clip"):
        self._properties = properties
        self._metadata = metadata or {}
        self._name = name

    def GetClipProperty(self, prop: str) -> str | None:
        return self._properties.get(prop)

    def GetMetadata(self, key: str) -> str | None:
        return self._metadata.get(key)

    def GetName(self) -> str:
        return self._name


class MockFolder:
    """Mock DaVinci Resolve folder object."""

    def __init__(self, clips: list | None = None, subfolders: list | None = None):
        self._clips = clips or []
        self._subfolders = subfolders or []

    def GetClipList(self):
        return self._clips

    def GetSubFolderList(self):
        return self._subfolders


def test_resolve_path() -> None:
    """resolve_path should return the file path from a clip."""
    clip = MockClip({"File Path": "/tmp/video.mp4"})
    assert manifest.resolve_path(clip) == "/tmp/video.mp4"

    clip_no_path = MockClip({})
    assert manifest.resolve_path(clip_no_path) == ""


def test_resolve_metadata() -> None:
    """resolve_metadata should return metadata value from a clip."""
    clip = MockClip({}, {"Keywords": "turtle,shark"})
    assert manifest.resolve_metadata(clip, "Keywords") == "turtle,shark"

    clip_no_meta = MockClip({}, {})
    assert manifest.resolve_metadata(clip_no_meta, "Keywords") == ""


def test_resolve_property() -> None:
    """resolve_property should return clip property value."""
    clip = MockClip({"FPS": "25.0", "Width": "1920"})
    assert manifest.resolve_property(clip, "FPS") == "25.0"
    assert manifest.resolve_property(clip, "Width") == "1920"
    assert manifest.resolve_property(clip, "Nonexistent") == ""


def test_extract_folder_no_clips() -> None:
    """extract_folder should return empty list for folder with no clips."""
    folder = MockFolder()
    result = manifest.extract_folder(folder, "Test Project", {"from_prefix": "", "to_prefix": ""})
    assert result == []


def test_extract_folder_with_clips() -> None:
    """extract_folder should extract clips with keywords."""
    clip = MockClip(
        properties={
            "File Path": "/Volumes/ExtremeSSD/Madagascar/dive1.mp4",
            "File Name": "dive1.mp4",
            "Clip ID": "cid123",
            "FPS": "25.0",
            "Width": "1920",
            "Height": "1080",
            "Duration": "60.0",
            "Start TC": "00:00:00:00",
            "End TC": "00:01:00:00",
        },
        metadata={"Keywords": "turtle,pelagic"},
        name="dive1",
    )
    folder = MockFolder(clips=[clip])
    path_mapping = {"from_prefix": "/Volumes/ExtremeSSD/Madagascar", "to_prefix": "/media/data"}

    result = manifest.extract_folder(folder, "Test Project", path_mapping)

    assert len(result) == 1
    assert result[0]["file_name"] == "dive1.mp4"
    assert result[0]["file_path"] == "/media/data/dive1.mp4"  # path mapping applied
    assert result[0]["tags"] == ["turtle", "pelagic"]
    assert result[0]["fps"] == "25.0"
    assert result[0]["resolution"] == "1920x1080"


def test_extract_folder_skips_clips_without_keywords() -> None:
    """extract_folder should skip clips without keywords."""
    clip_no_keywords = MockClip(
        properties={"File Path": "/tmp/dive1.mp4", "File Name": "dive1.mp4"},
        metadata={},
        name="dive1",
    )
    clip_with_keywords = MockClip(
        properties={"File Path": "/tmp/dive2.mp4", "File Name": "dive2.mp4"},
        metadata={"Keywords": "shark"},
        name="dive2",
    )
    folder = MockFolder(clips=[clip_no_keywords, clip_with_keywords])

    result = manifest.extract_folder(folder, "Test Project", {"from_prefix": "", "to_prefix": ""})

    assert len(result) == 1
    assert result[0]["file_name"] == "dive2.mp4"


def test_extract_folder_recursive() -> None:
    """extract_folder should recurse into subfolders."""
    clip = MockClip(
        properties={"File Path": "/tmp/deep.mp4", "File Name": "deep.mp4"},
        metadata={"Keywords": "macro"},
        name="deep",
    )
    subfolder = MockFolder(clips=[clip])
    folder = MockFolder(subfolders=[subfolder])

    result = manifest.extract_folder(folder, "Test Project", {"from_prefix": "", "to_prefix": ""})

    assert len(result) == 1
    assert result[0]["file_name"] == "deep.mp4"


def test_get_resolve_not_available(monkeypatch) -> None:
    """get_resolve should return None when DaVinciResolveScript is not available."""
    # Block the import
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "DaVinciResolveScript":
            raise ImportError("No module named 'DaVinciResolveScript'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    result = manifest.get_resolve()
    assert result is None


def test_clip_id_hash_fallback() -> None:
    """clip_id should fall back to hash when Clip ID is empty."""
    clip = MockClip(
        properties={
            "File Path": "/tmp/dive.mp4",
            "File Name": "dive.mp4",
            # No Clip ID
            "FPS": "25.0",
        },
        metadata={"Keywords": "turtle"},
        name="dive",
    )
    folder = MockFolder(clips=[clip])
    result = manifest.extract_folder(folder, "Test Project", {"from_prefix": "", "to_prefix": ""})

    assert len(result) == 1
    assert result[0]["clip_id"] != ""  # Should have a hash-based ID
    assert len(result[0]["clip_id"]) == 16  # SHA256 truncated to 16 chars
