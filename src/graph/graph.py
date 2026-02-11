from pathlib import Path
from typing import List, Type
from typing import Generic, TypeVar, List
import base64

from pydantic import BaseModel, field_serializer, Field
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from type import PDFInput, PageRange
from annotator.pdf_annotator import PDFAnnotator
from pdf_llm.pdf_llm import PDFMultiModalLLM
from pdf_seperator.pdf_seperator import PDFSeperator
from pdf_image_converter import PDFImageConverter

load_dotenv()


class Section(BaseModel):
    """
    Base class for any structured unit extracted from a document.
    This should contain ONLY semantic fields produced by the LLM.
    """

    page_range: PageRange


T = TypeVar("T", bound=Section)


class ListOutput(BaseModel, Generic[T]):
    items: List[T]


class ParsedUnit(
    BaseModel,
    Generic[T],
):
    """
    A semantic unit enriched with pipeline-generated artifacts.
    """

    data: T
    pdf_bytes: bytes | None = None

    @field_serializer("pdf_bytes")
    def serialize_pdf_bytes(self, value: bytes):
        return base64.b64encode(value).decode("ascii")


model = init_chat_model(model="gpt-4o", model_provider="openai")


class State(BaseModel, Generic[T]):
    # --- Inputs ---
    pdf: str | Path
    prompt: str
    pdf_bytes: bytes | None = None

    # --- Schema configuration ---
    output_schema: Type[ListOutput[T]] = Field(exclude=True)
    raw_output: List[T] = []

    parsed: list[ParsedUnit[T]] = Field(default_factory=list, exclude=False)

    @field_serializer("pdf_bytes")
    def serialize_pdf_bytes(self, value: bytes):
        return base64.b64encode(value).decode("ascii")


def prepare_pdf(state: State):
    state.pdf = Path(state.pdf)
    if not state.pdf.exists():
        raise ValueError("PDF path cannot be resolved")
    pdf = PDFAnnotator(state.pdf).annotate_and_render_pages()
    return {"pdf_bytes": pdf}


def get_sections(state: State):
    llm = PDFMultiModalLLM(
        prompt=state.prompt,
        pdf=state.pdf_bytes,
        model=model,
    )
    result = llm.invoke(state.output_schema)
    result = state.output_schema.model_validate(result)
    return {"raw_output": result.items}


def seperate_pages(state: State[T]):
    parsed = []
    if not state.pdf_bytes:
        raise ValueError("PDF bytes is None")
    separator = PDFSeperator(pdf_bytes=state.pdf_bytes)
    for unit in state.raw_output:
        page_range = getattr(unit, "page_range", None)
        if page_range is None:
            raise ValueError("Unit does not define a page_range")
        cleaned = ParsedUnit[T](
            data=unit,
            pdf_bytes=separator.extract_page_range(
                start=page_range.start_page - 1,
                end=page_range.end_page - 1,
            ),
        )
        parsed.append(cleaned)
    return {"parsed": parsed}


graph = StateGraph(State)
graph.add_node(prepare_pdf)
graph.add_node(get_sections)
graph.add_node(seperate_pages)

graph.add_edge(START, "prepare_pdf")
graph.add_edge("prepare_pdf", "get_sections")
graph.add_edge("get_sections", "seperate_pages")
graph.add_edge("seperate_pages", END)
graph = graph.compile()


if __name__ == "__main__":
    path = "data/Lecture_02_03.pdf"
    from typing import Literal

    class MySection(Section, BaseModel):
        title: str
        description: str
        section_type: Literal["derivation", "question"]

    class MySections(ListOutput[MySection]):
        items: List[MySection]

    result = graph.invoke(
        State(
            output_schema=MySections,
            pdf=path,
            prompt="""You are analyzing a set of lecture notes.

Your task is to identify and extract distinct sections that fall into ONE of the following two categories only:

1. **Derivation**
   - A derivation is a mathematical development that proceeds step-by-step using equations, formulas, algebra, calculus, or symbolic manipulation.
   - It typically starts from assumptions, definitions, or governing equations and arrives at a derived result.
   - Only include content that is explicitly part of the mathematical derivation.
   - Do NOT explain, summarize, or add interpretation beyond what is written.

2. **Question**
   - A question is a clearly defined problem or practice exercise posed to the reader.
   - It may begin with phrases such as “Find”, “Determine”, “Calculate”, “Show that”, or be labeled as an example, problem, or practice question.
   - Only include the question statement itself.
   - Do NOT include solution steps unless they are explicitly written as part of the question.

For each identified section:
- Create a separate section entry.
- Assign the appropriate `section_type` (`"derivation"` or `"question"`).
- Use the section’s visible heading or a concise descriptive title.
- Provide a short description that closely reflects the original content without adding new information.

Page indexing:
- The lecture pages are annotated with a circled page number in the bottom-left corner.
- Use this circled page number as the authoritative reference when determining where a section begins and ends.

Important constraints:
- Do not merge multiple derivations or questions into a single section.
- Do not invent structure that is not present in the lecture.
- If content is ambiguous, only include it if it clearly fits one of the two categories.""",
        )
    )
    result = State.model_validate(result)
    from utils import to_serializable
    import json

    output = Path("output.json").resolve()
    output.write_text(json.dumps(to_serializable(result)))
