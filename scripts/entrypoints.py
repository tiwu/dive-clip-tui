"""Console entry points for scripts whose filenames start with a number."""

from importlib import import_module


def _run(script: str) -> None:
    import_module(f".{script}", package=__package__).main()


def scan_footage() -> None:
    _run("02_scan_footage")


def generate_edl() -> None:
    _run("03_generate_edl")


def plan_storyline() -> None:
    _run("04_plan_storyline")


def ingest_media() -> None:
    _run("00_ingest_media")


def render_clips() -> None:
    _run("05_render_clips")


def assemble_clips() -> None:
    _run("06_assemble_video")


def create_aftermovie() -> None:
    _run("07_create_aftermovie")
