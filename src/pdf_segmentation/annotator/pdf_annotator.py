from pathlib import Path
from typing import Literal, Optional, Tuple


import pymupdf
from pymupdf import Page


from pdf_invoke.converter import PDFImageConverter
from pdf_segmentation.types import Anchor, AnchorPos


class PDFAnnotator:
    def __init__(
        self,
        pdf_path: str | Path,
        anchor: Anchor | AnchorPos = Anchor.BOTTOM_LEFT,
        margin_frac: float = 1 / 10,
        offset: Tuple[int, int] = (10, 10),
        zoom: float = 2.0,
    ):
        self.pdf = Path(pdf_path).resolve()
        self.anchor = Anchor(anchor)
        self.margin_frac = margin_frac
        self.offset = offset
        self.zoom = zoom
        self._validate()

    def annotate_and_render_pages(
        self,
    ) -> bytes:
        doc = pymupdf.open(self.pdf)
        try:
            # Annotate all pages
            for page in doc:
                assert isinstance(page, Page)
                self._annotate_page(page)
            # Return annotated PDF as bytes
            return doc.tobytes()
        finally:
            doc.close()

    def _annotate_and_save(
        self,
        name: str,
        output_path: str | Path,
        method: Literal["pdf", "image"] = "image",
    ) -> str:
        data = self.annotate_and_render_pages()
        converter = PDFImageConverter()
        output_path = Path(output_path)
        if method == "image":
            converter.save_pdf_to_images(data, output_path=output_path, pdf_name=name)
        elif method == "pdf":
            filepath = output / name
            if filepath.suffix.lower() != ".pdf":
                filepath = filepath.with_suffix(".pdf")
            filepath.write_bytes(data)
            return filepath.as_posix()
        else:
            raise TypeError(f"Method {method} is not valid for annotating")
        return output_path.as_posix()

    def _annotate_page(
        self,
        page: Page,
    ):
        if page.rotation != 0:
            page.set_rotation(0)

        rect = page.rect
        cx, cy = self._get_annotation_coords(
            (rect.width, rect.height),
            self.margin_frac,
            self.offset,
            self.anchor,
        )

        radius = rect.width * self.margin_frac

        # draw circle
        page.draw_circle(
            center=(cx, cy),
            radius=radius,
        )

        # draw centered number
        box_size = radius
        label_rect = pymupdf.Rect(
            cx - box_size,
            cy - box_size,
            cx + box_size,
            cy + box_size,
        )

        page.insert_textbox(
            label_rect,
            str(page.number),
            fontsize=radius,
            align=pymupdf.TEXT_ALIGN_CENTER,
        )

    def _get_annotation_coords(
        self,
        size: tuple[int, int],
        margin_frac: float = 1 / 10,
        offset: tuple[int, int] = (10, 10),
        anchor: Anchor = Anchor.BOTTOM_LEFT,
    ) -> Tuple[float, float]:
        width, height = size
        x_off, y_off = offset

        match anchor.value:
            case "top-left":
                cx = width * margin_frac + x_off
                cy = height * margin_frac + y_off
            case "top-right":
                cx = width - (width * margin_frac) - x_off
                cy = height * margin_frac + y_off
            case "bottom-left":
                cx = width * margin_frac + x_off
                cy = height - (height * margin_frac) - y_off
            case "bottom-right":
                cx = width - (width * margin_frac) - x_off
                cy = height - (height * margin_frac) - y_off
            case _:
                raise ValueError(f"Invalid anchor: {anchor}")
        return cx, cy

    def _validate(self):
        if not self.pdf.exists():
            raise FileNotFoundError(f"PDF Path {self.pdf} does not exist")


if __name__ == "__main__":
    path = Path("data/Lecture_02_03.pdf")
    output = Path(r"data\images").resolve()
    name = path.stem + "_test"
    PDFAnnotator(path, anchor="bottom-left", margin_frac=1 / 20)._annotate_and_save(
        method="pdf", output_path=output, name=name
    )
