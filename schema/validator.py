from pydantic import BaseModel,Field
from typing import Literal


# Document relevance validation model
class GradeDocument(BaseModel):
    """verifies the retrieved documents is relevant to user's question."""
    score:Literal["Yes","No"] = Field(
        ...,
        description="Is document relevant to user's question? If yes ->'Yes' ,if no ->'No'"
    )