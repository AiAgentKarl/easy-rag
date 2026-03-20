"""
Retriever — Findet die relevantesten Chunks für eine Anfrage.

Nutzt den TF-IDF-Index aus embedder.py für die Ähnlichkeitsberechnung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from easy_rag.chunker import Chunk
from easy_rag.embedder import TFIDFIndex, cosine_similarity


@dataclass
class SearchResult:
    """Ein Suchergebnis mit Chunk, Score und Metadaten."""

    chunk: Chunk
    score: float  # Kosinus-Ähnlichkeit (0-1)
    rank: int  # Rang im Ergebnis (1-basiert)

    def __repr__(self) -> str:
        preview = self.chunk.text[:80].replace("\n", " ")
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.3f}, "
            f"source='{self.chunk.source}', text='{preview}...')"
        )


class Retriever:
    """
    Verwaltet Chunks und findet die relevantesten für eine Suchanfrage.

    Intern wird ein TF-IDF-Index aufgebaut. Die Suche gibt die
    Top-K ähnlichsten Chunks mit Score und Metadaten zurück.
    """

    def __init__(self):
        self.index = TFIDFIndex()
        self.chunks: List[Chunk] = []
        self._sources: Dict[str, int] = {}  # Quelle -> Anzahl Chunks

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Fügt eine Liste von Chunks zum Index hinzu."""
        for chunk in chunks:
            self.index.add_document(chunk.text)
            self.chunks.append(chunk)

            # Quellen-Statistik aktualisieren
            source = chunk.source
            self._sources[source] = self._sources.get(source, 0) + 1

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Sucht die relevantesten Chunks für eine Anfrage.

        Args:
            query: Suchanfrage als Text
            top_k: Maximale Anzahl der Ergebnisse

        Returns:
            Liste von SearchResult-Objekten, sortiert nach Relevanz
        """
        if not self.chunks:
            return []

        # Query-Vektor berechnen
        query_vec = self.index.query_vector(query)

        if not query_vec:
            return []

        # Ähnlichkeit mit allen Chunks berechnen
        scores = []
        for idx, chunk in enumerate(self.chunks):
            doc_vec = self.index.get_doc_vector(idx)
            score = cosine_similarity(query_vec, doc_vec)
            if score > 0:
                scores.append((idx, score))

        # Nach Score absteigend sortieren
        scores.sort(key=lambda x: x[1], reverse=True)

        # Top-K Ergebnisse zurückgeben
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=round(score, 4),
                    rank=rank,
                )
            )

        return results

    @property
    def num_chunks(self) -> int:
        """Gesamtzahl der indexierten Chunks."""
        return len(self.chunks)

    @property
    def num_sources(self) -> int:
        """Anzahl der verschiedenen Quelldokumente."""
        return len(self._sources)

    @property
    def sources(self) -> Dict[str, int]:
        """Übersicht: Quelle -> Anzahl Chunks."""
        return dict(self._sources)

    def stats(self) -> Dict:
        """Gibt Statistiken über den Index zurück."""
        return {
            "num_chunks": self.num_chunks,
            "num_sources": self.num_sources,
            "num_unique_terms": len(self.index.doc_freq),
            "sources": self.sources,
        }
