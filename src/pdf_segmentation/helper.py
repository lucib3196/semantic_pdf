# from pathlib import Path
# from graph.graph import State
# import json
# from utils import save_base64_image
# path = Path("output.json").resolve()
# data = State.model_validate(json.loads(path.read_text()))
# save_base64_image(data.parsed[-1].pdf_bytes, "data_output.pdf")

from langchain.chat_models import init_chat_model
from pdf_invoke import MultiModalLLM
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

llm = MultiModalLLM(
    prompt="What is in the image",
    model=init_chat_model(model="gpt-4o", model_provider="openai"),
)
data = Path(r"data\Lecture_02_03.pdf")
result = llm.invoke(pdf=data)
print(result)
