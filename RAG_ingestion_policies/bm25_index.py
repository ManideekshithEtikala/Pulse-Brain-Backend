# app/agents/brain_agents/policy_agent/nodes/bm25_retriever.py
import json
from pathlib import Path

from rank_bm25 import BM25Okapi


def find_chunks_dir() -> Path:
    """Walk up from this file until we hit a 'chunks' dir holding chunk files."""
    here = Path(__file__).resolve()
    for root in [here] + list(here.parents):
        candidate = root / "chunks"
        if candidate.is_dir() and any(candidate.glob("*_chunks.json")):
            return candidate
    raise FileNotFoundError("No 'chunks' dir with *_chunks.json above " + str(here))


class BM25Retriever:
    def __init__(self, chunks_dir: Path | None = None):
        self.chunks_dir = chunks_dir or find_chunks_dir()

        self.chunks: list[dict] = []
        for path in sorted(self.chunks_dir.glob("*_chunks.json")):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "chunks" not in data:
                print(f"[skip] {path.name}: not a chunks file")
                continue
            self.chunks.extend(data["chunks"])
            print(f"  loaded {path.name}: {len(data['chunks'])} chunks")

        if not self.chunks:
            raise RuntimeError(f"No chunks loaded from {self.chunks_dir}")

        ids = [c["id"] for c in self.chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs across chunk files!"

        self.tokenized = [self.tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"BM25 index built: {len(self.chunks)} chunks from {self.chunks_dir}")

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(self.tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for rank, idx in enumerate(ranked[:top_k], start=1):
            chunk = self.chunks[idx]
            meta = chunk.get("metadata", {})
            results.append(
                {
                    "id": chunk["id"],
                    "rank": rank,
                    "bm25_score": float(scores[idx]),
                    "policy_id": meta.get("policy_id", ""),
                    "section": meta.get("section_number", "?"),
                    "chunk_type": meta.get("chunk_type", ""),
                    "text": chunk["text"],
                    "metadata": meta,
                }
            )
        return results
