from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 0.0

    def to_dict(self): return asdict(self)

@dataclass
class CutInterval:
    start: float
    end: float
    reason: str
    confidence: float = 1.0
    detail: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    word_indices: list[int] = field(default_factory=list)

    def to_dict(self): return asdict(self)

@dataclass
class KeepInterval:
    start: float
    end: float
    reason: str = "speech"

    def to_dict(self): return asdict(self)
