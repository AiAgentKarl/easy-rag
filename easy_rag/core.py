"""
EasyRAG — Die Hauptklasse. RAG in 3 Zeilen Code.

Beispiel:
    from easy_rag import EasyRAG
    rag = EasyRAG("./meine_dokumente/")
    antwort = rag.ask("Was steht im Vertrag über Kündigung?")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from easy_rag.chunker import Chunk, load_documents, read_file, chunk_text
from easy_rag.retriever import Retriever, SearchResult


@dataclass
class RAGResult:
    """Strukturiertes Ergebnis einer RAG-Anfrage."""

    query: str
    chunks: List[SearchResult]
    answer: str  # Zusammengesetzte Antwort aus den relevantesten Chunks

    def __repr__(self) -> str:
        return (
            f"RAGResult(query='{self.query}', "
            f"num_results={len(self.chunks)}, "
            f"top_score={self.chunks[0].score if self.chunks else 0})"
        )

    def __str__(self) -> str:
        return self.answer


class EasyRAG:
    """
    RAG in 3 Zeilen Code — Lade Dokumente, stelle Fragen, bekomme Antworten.

    Keine externen Abhängigkeiten für die Basisversion nötig.
    Nutzt TF-IDF für die Textsuche (eingebaut in Python-Standardbibliothek).

    Beispiel:
        from easy_rag import EasyRAG
        rag = EasyRAG("./docs/")
        result = rag.ask("What does the contract say about termination?")
        print(result)

    Unterstützte Formate: .txt, .md, .json (+ .pdf mit pip install easy-rag[pdf])
    """

    def __init__(
        self,
        path: Optional[str] = None,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        """
        Erstellt einen neuen EasyRAG-Index.

        Args:
            path: Verzeichnis oder Datei zum Laden. None = leerer Index.
            chunk_size: Maximale Zeichenanzahl pro Chunk (Standard: 500)
            overlap: Überlappung zwischen Chunks (Standard: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retriever = Retriever()
        self._loaded_paths: List[str] = []

        if path is not None:
            self._load_path(path)

    def _load_path(self, path: str) -> int:
        """
        Lädt Dokumente aus einem Pfad.

        Returns:
            Anzahl der geladenen Chunks
        """
        chunks = load_documents(path, self.chunk_size, self.overlap)
        self.retriever.add_chunks(chunks)
        self._loaded_paths.append(str(path))
        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> RAGResult:
        """
        Stelle eine Frage an deine Dokumente.

        Findet die relevantesten Text-Chunks und gibt sie
        als strukturiertes Ergebnis zurück.

        Args:
            question: Die Frage als natürlicher Text
            top_k: Anzahl der relevantesten Chunks (Standard: 3)

        Returns:
            RAGResult mit Antwort-Chunks und formatierter Antwort
        """
        results = self.retriever.search(question, top_k=top_k)

        # Antwort aus den besten Chunks zusammenbauen
        if not results:
            answer = "Keine relevanten Informationen gefunden."
        else:
            parts = []
            for r in results:
                source_name = Path(r.chunk.source).name if r.chunk.source else "unknown"
                parts.append(
                    f"[{source_name} | Score: {r.score:.2f}]\n{r.chunk.text}"
                )
            answer = "\n\n---\n\n".join(parts)

        return RAGResult(
            query=question,
            chunks=results,
            answer=answer,
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Sucht relevante Chunks ohne formatierte Antwort.

        Nützlich wenn du die rohen Ergebnisse weiterverarbeiten willst.

        Args:
            query: Suchanfrage als Text
            top_k: Maximale Anzahl der Ergebnisse (Standard: 5)

        Returns:
            Liste von SearchResult-Objekten
        """
        return self.retriever.search(query, top_k=top_k)

    def add_document(self, filepath: str) -> int:
        """
        Fügt ein einzelnes Dokument zum Index hinzu.

        Args:
            filepath: Pfad zur Datei

        Returns:
            Anzahl der neuen Chunks
        """
        text = read_file(filepath)
        chunks = chunk_text(
            text, self.chunk_size, self.overlap, source=str(filepath)
        )
        self.retriever.add_chunks(chunks)
        return len(chunks)

    def add_text(self, text: str, source: str = "inline") -> int:
        """
        Fügt rohen Text direkt zum Index hinzu (ohne Datei).

        Args:
            text: Der Text zum Indexieren
            source: Quellenbezeichnung (Standard: "inline")

        Returns:
            Anzahl der neuen Chunks
        """
        chunks = chunk_text(
            text, self.chunk_size, self.overlap, source=source
        )
        self.retriever.add_chunks(chunks)
        return len(chunks)

    def stats(self) -> Dict:
        """
        Gibt Statistiken über den Index zurück.

        Returns:
            Dict mit num_chunks, num_sources, num_unique_terms, sources
        """
        stats = self.retriever.stats()
        stats["chunk_size"] = self.chunk_size
        stats["overlap"] = self.overlap
        stats["loaded_paths"] = self._loaded_paths
        return stats

    def __repr__(self) -> str:
        return (
            f"EasyRAG(chunks={self.retriever.num_chunks}, "
            f"sources={self.retriever.num_sources})"
        )

    def __len__(self) -> int:
        return self.retriever.num_chunks
