from typing import Literal

_orientation = Literal["left", "right"]


def laterality_to_orientation(lat: str) -> _orientation:
    """Convert a laterality code to an orientation string.

    Args:
        lat (str): Laterality code — ``"L"`` or ``"R"`` (case-insensitive).

    Raises:
        ValueError: If ``lat`` is not ``"L"`` or ``"R"``.

    Returns:
        _orientation: ``"left"`` for ``"L"``, ``"right"`` for ``"R"``.
    """
    lat = lat.strip().upper()
    if lat == "L":
        return "left"
    elif lat == "R":
        return "right"
    raise ValueError(f"Invalid laterality '{lat}'. Expected 'L' or 'R'.")


def orientation_to_laterality(orientation: _orientation) -> str:
    """Convert an orientation string to a laterality code.

    Args:
        orientation (_orientation): ``"left"`` or ``"right"``.

    Raises:
        ValueError: If ``orientation`` is not ``"left"`` or ``"right"``.

    Returns:
        str: ``"L"`` for ``"left"``, ``"R"`` for ``"right"``.
    """
    orientation = orientation.strip().lower()

    if orientation == "left":
        return "L"
    elif orientation == "right":
        return "R"
    raise ValueError(
        f"Invalid orientation '{orientation}'. Expected 'left' or 'right'."
    )
