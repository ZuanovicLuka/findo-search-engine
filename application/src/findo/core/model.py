from typing import Optional

from pydantic import BaseModel


class Document(BaseModel):
    """Represents a single document in the search results"""

    id: int
    title: str
    content: str
    snippet: Optional[str] = None
