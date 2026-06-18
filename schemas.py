from pydantic import BaseModel

class ConceptSummary(BaseModel):
    title: str
    summary: str
    difficulty: str