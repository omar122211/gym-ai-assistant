"""Knowledge base: split the markdown files into sections, embed them, store them in Chroma.

Run `python -m app.knowledge` to (re)build the index.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.providers import AIProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    section: str
    text: str


def split_markdown(path: Path) -> list[Chunk]:
    """One chunk per '## ' section; the document title is repeated in each chunk for context."""
    raw = path.read_text(encoding="utf-8").strip()
    title_match = re.match(r"#\s+(.+)", raw)
    title = title_match.group(1).strip() if title_match else path.stem
    body = raw[title_match.end():] if title_match else raw

    parts = re.split(r"(?m)^##\s+", body)
    chunks: list[Chunk] = []
    intro = parts[0].strip()
    if intro:
        chunks.append(_chunk(path, title, title, intro))
    for part in parts[1:]:
        heading, _, text = part.partition("\n")
        if text.strip():
            chunks.append(_chunk(path, title, heading.strip(), text.strip()))
    return chunks


def _chunk(path: Path, title: str, section: str, text: str) -> Chunk:
    content = f"{title} — {section}\n{text}" if section != title else f"{title}\n{text}"
    cid = hashlib.sha1(f"{path.name}|{section}".encode()).hexdigest()[:16]
    return Chunk(id=cid, file=path.name, section=section, text=content)


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        chunks.extend(split_markdown(path))
    return chunks


class KnowledgeBase:
    def __init__(self, chroma_dir: Path, collection: str):
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._manifest = chroma_dir / f"{collection}.manifest.json"

    def count(self) -> int:
        return self.collection.count()

    def build(self, chunks: list[Chunk], provider: AIProvider, force: bool = False) -> int:
        """Embed and store chunks. Skips work when nothing changed (saves API quota)."""
        fingerprint = hashlib.sha1(json.dumps([[c.id, c.text] for c in chunks]).encode()).hexdigest()
        if not force and self._manifest.exists() and self.count() == len(chunks):
            if json.loads(self._manifest.read_text()).get("fingerprint") == fingerprint:
                log.info("knowledge base unchanged", extra={"fields": {"chunks": len(chunks)}})
                return 0

        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        vectors = provider.embed([c.text for c in chunks], kind="document")
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"file": c.file, "section": c.section} for c in chunks],
        )
        self._manifest.write_text(json.dumps({"fingerprint": fingerprint, "provider": provider.name}))
        log.info("knowledge base built", extra={"fields": {"chunks": len(chunks), "provider": provider.name}})
        return len(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        if self.count() == 0:
            return []
        res = self.collection.query(query_embeddings=[query_vector], n_results=min(top_k, self.count()))
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"text": doc, "file": meta["file"], "section": meta["section"], "similarity": round(1 - dist, 4)})
        return hits


if __name__ == "__main__":
    from app.config import get_settings
    from app.logging_setup import setup_logging
    from app.providers import build_provider

    s = get_settings()
    setup_logging(s.log_level)
    kb = KnowledgeBase(s.chroma_dir, s.collection)
    n = kb.build(load_chunks(s.knowledge_dir), build_provider(s), force="--force" in __import__("sys").argv)
    print(f"Indexed {n} chunks" if n else "Index already up to date", f"({kb.count()} in store)")
