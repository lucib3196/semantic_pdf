# PDF Segmentation

**PDF Segmentation** is a utility for parsing PDFs into **structured page-range chunks** using **LLMs**, designed for downstream processing such as image conversion, question extraction, derivations, and document analysis.

It is intended to sit at the **front of a PDF processing pipeline**, handling segmentation and orchestration.

---

## Features

* LLM-driven PDF segmentation into page ranges
* Schema-validated, structured outputs
* PDF → image conversion utilities
* PDF page and range splitting
* Multimodal (PDF + image) LLM support

---

## Core Components

### Segmentation Graph

```python
from graph.graph import (
    graph as PDFSegmentation,
    Section,
    ListOutput,
    State as SegmentationInput,
)
```

* **`PDFSegmentation`** – main graph entry point
* **`Section`** – base class for output units
* **`ListOutput[T]`** – typed LLM output container
* **`State`** – input configuration for segmentation

The graph:

* Accepts a PDF (path or bytes)
* Invokes an LLM
* Returns schema-validated segmentation results

---

### PDF Utilities

**PDF → Images**

```python
from pdf_image_converter import PDFImageConverter
```

**PDF Page Splitting**

```python
from pdf_seperator import PDFSeperator
```

**Multimodal LLM Invocation**

```python
from pdf_llm import PDFMultiModalLLM
```

---

## Example

### Define an Output Schema

```python
from typing import Literal, List
from pydantic import BaseModel
from graph.graph import Section, ListOutput

class MySection(Section, BaseModel):
    title: str
    description: str
    section_type: Literal["derivation", "question"]

class MySections(ListOutput[MySection]):
    items: List[MySection]
```

---

### Run Segmentation

```python
from pathlib import Path
from graph.graph import graph, State

result = graph.invoke(
    State(
        pdf=Path("data/Lecture_02_03.pdf"),
        prompt="Extract the content into these chunks",
        output_schema=MySections,
    )
)
```

---

## Input State

```python
class State(BaseModel, Generic[T]):
    pdf: str | Path
    prompt: str
    pdf_bytes: bytes | None = None

    output_schema: Type[ListOutput[T]] = Field(exclude=True)
    raw_output: List[T] = []
    parsed: list[ParsedUnit[T]] = Field(default_factory=list)

    @field_serializer("pdf_bytes")
    def serialize_pdf_bytes(self, value: bytes):
        return base64.b64encode(value).decode("ascii")
```

**Notes**

* Provide either `pdf` or `pdf_bytes`
* `output_schema` controls the LLM output structure
* `parsed` contains validated segmentation results

---

## Typical Pipeline

```text
PDF → LLM Segmentation → Page Ranges
   → PDF Splitter
   → PDF → Image Conversion
   → Downstream Processing
```

Common use cases include lecture parsing, educational content generation, and document analysis.

---
