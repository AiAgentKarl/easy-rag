"""
TF-IDF Embedding — Vektorisierung ohne externe Abhängigkeiten.

Nutzt nur Python-Standardbibliothek (collections.Counter, math).
Damit funktioniert die Basisversion komplett ohne pip install.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Tuple


# Stoppwörter für bessere Relevanz (Englisch + Deutsch)
STOP_WORDS = {
    # Englisch
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "above", "after", "again", "also", "any", "as", "because",
    "before", "between", "down", "during", "if", "into", "like", "much",
    "new", "now", "off", "once", "out", "over", "own", "re", "then",
    "there", "through", "under", "up", "well",
    # Deutsch
    "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in", "auf",
    "an", "zu", "für", "von", "mit", "bei", "aus", "ist", "sind", "war",
    "hat", "haben", "wird", "werden", "kann", "können", "ich", "du", "er",
    "sie", "es", "wir", "ihr", "nicht", "auch", "noch", "nur", "schon",
    "dann", "wenn", "als", "so", "wie", "was", "wer", "wo", "den", "dem",
    "des", "dass", "über", "nach", "vor", "um", "sich", "sein", "seine",
    "ihre", "einem", "einen", "einer", "eines", "diese", "dieser", "dieses",
}


def tokenize(text: str) -> List[str]:
    """
    Zerlegt Text in normalisierte Tokens.

    - Kleinschreibung
    - Satzzeichen entfernen
    - Stoppwörter filtern
    - Tokens kürzer als 2 Zeichen ignorieren
    """
    # Kleinschreibung und Satzzeichen entfernen
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    # In Wörter aufteilen
    words = text.split()

    # Filtern: Stoppwörter und zu kurze Wörter entfernen
    tokens = [w for w in words if w not in STOP_WORDS and len(w) >= 2]

    return tokens


class TFIDFIndex:
    """
    TF-IDF Index — berechnet Term Frequency-Inverse Document Frequency.

    Komplett ohne externe Abhängigkeiten implementiert.
    """

    def __init__(self):
        # Token-Frequenzen pro Dokument
        self.doc_term_freqs: List[Counter] = []
        # Anzahl der Dokumente, die jeden Term enthalten
        self.doc_freq: Counter = Counter()
        # Gesamtzahl der Dokumente
        self.num_docs: int = 0
        # IDF-Werte (werden bei Bedarf berechnet)
        self._idf_cache: Dict[str, float] = {}
        self._dirty: bool = True

    def add_document(self, text: str) -> int:
        """
        Fügt ein Dokument zum Index hinzu.

        Returns:
            Index des hinzugefügten Dokuments
        """
        tokens = tokenize(text)
        term_freq = Counter(tokens)

        doc_idx = self.num_docs
        self.doc_term_freqs.append(term_freq)

        # Dokument-Frequenz aktualisieren (jeder Term zählt nur 1x pro Dokument)
        for term in set(tokens):
            self.doc_freq[term] += 1

        self.num_docs += 1
        self._dirty = True

        return doc_idx

    def _compute_idf(self) -> None:
        """Berechnet IDF-Werte für alle bekannten Terme."""
        self._idf_cache = {}
        for term, df in self.doc_freq.items():
            # IDF mit Smoothing: log((N + 1) / (df + 1)) + 1
            self._idf_cache[term] = math.log((self.num_docs + 1) / (df + 1)) + 1
        self._dirty = False

    def get_tfidf_vector(self, term_freq: Counter) -> Dict[str, float]:
        """
        Berechnet den TF-IDF-Vektor für gegebene Term-Frequenzen.

        TF = Anzahl des Terms / Gesamtanzahl der Tokens
        IDF = log((N + 1) / (df + 1)) + 1
        TF-IDF = TF * IDF
        """
        if self._dirty:
            self._compute_idf()

        total_terms = sum(term_freq.values())
        if total_terms == 0:
            return {}

        vector = {}
        for term, count in term_freq.items():
            tf = count / total_terms
            idf = self._idf_cache.get(term, math.log(self.num_docs + 1) + 1)
            vector[term] = tf * idf

        return vector

    def get_doc_vector(self, doc_idx: int) -> Dict[str, float]:
        """Gibt den TF-IDF-Vektor eines indexierten Dokuments zurück."""
        if doc_idx < 0 or doc_idx >= self.num_docs:
            raise IndexError(f"Dokument-Index {doc_idx} ungültig")
        return self.get_tfidf_vector(self.doc_term_freqs[doc_idx])

    def query_vector(self, query: str) -> Dict[str, float]:
        """Berechnet den TF-IDF-Vektor für eine Suchanfrage."""
        tokens = tokenize(query)
        term_freq = Counter(tokens)
        return self.get_tfidf_vector(term_freq)


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """
    Berechnet die Kosinus-Ähnlichkeit zwischen zwei Sparse-Vektoren.

    Werte zwischen 0 (keine Ähnlichkeit) und 1 (identisch).
    """
    if not vec_a or not vec_b:
        return 0.0

    # Schnittmenge der Terme finden
    common_terms = set(vec_a.keys()) & set(vec_b.keys())

    if not common_terms:
        return 0.0

    # Skalarprodukt berechnen
    dot_product = sum(vec_a[term] * vec_b[term] for term in common_terms)

    # Normen berechnen
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
