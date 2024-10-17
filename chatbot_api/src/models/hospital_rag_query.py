from pydantic import BaseModel


class CVDQueryInput(BaseModel):
    text: str


class CVDQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
