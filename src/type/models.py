from pydantic import BaseModel


class BaseOutput(BaseModel):
    data: str
