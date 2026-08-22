import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyChunk:
    source: str
    text: str
    score: int


class PolicyIndex:
    """Dependency-free lexical baseline used before introducing dense retrieval."""

    def __init__(self, policy_dir: Path) -> None:
        self.policy_dir = policy_dir
        self._chunks = self._load()

    def _load(self) -> list[tuple[str, str, set[str]]]:
        chunks: list[tuple[str, str, set[str]]] = []
        for path in sorted(self.policy_dir.glob("*.md")):
            for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                text = paragraph.strip()
                if text:
                    chunks.append((path.name, text, self._tokens(text)))
        return chunks

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def search(self, query: str, limit: int = 4) -> list[PolicyChunk]:
        query_tokens = self._tokens(query)
        ranked = [
            PolicyChunk(source=source, text=text, score=len(query_tokens & tokens))
            for source, text, tokens in self._chunks
        ]
        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        positive = [chunk for chunk in ranked if chunk.score > 0]
        return (positive or ranked)[:limit]

    def context(self, query: str, limit: int = 4) -> str:
        return "\n\n".join(
            f"[Policy: {chunk.source}]\n{chunk.text}" for chunk in self.search(query, limit)
        )
