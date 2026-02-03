from pathlib import Path
from typing import Literal
from enum import Enum


AnchorPos = Literal["top-left", "top-right", "bottom-right", "bottom-left"]


class Anchor(str, Enum):
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"


PDFInput = str | Path
ImageBytes = bytes

ImageInput = PDFInput | ImageBytes
