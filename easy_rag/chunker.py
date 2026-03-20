"""
Dokument-Chunking — Texte intelligent in Stücke zerlegen.

Unterstützt .txt, .md, .json Dateien direkt.
Für .pdf ist pdfplumber als optionale Abhängigkeit nötig.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Unterstützte Dateiendungen
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


@dataclass
class Chunk:
    """Ein einzelnes Text-Stück mit Metadaten."""

    text: str
    source: str  # Dateipfad
    chunk_index: int  # Position im Dokument
    start_char: int  # Start-Position im Originaltext
    end_char: int  # End-Position im Originaltext
    metadata: dict = field(default_factory=dict)


def read_file(filepath: str) -> str:
    """
    Liest eine Datei und gibt den Text zurück.

    Unterstützt: .txt, .md, .json, .pdf (mit pdfplumber)
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return _read_text_file(path)
    elif ext == ".json":
        return _read_json_file(path)
    elif ext == ".pdf":
        return _read_pdf_file(path)
    else:
        # Versuche als Textdatei zu lesen
        return _read_text_file(path)


def _read_text_file(path: Path) -> str:
    """Liest eine einfache Textdatei."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Konnte Datei nicht lesen: {path}")


def _read_json_file(path: Path) -> str:
    """
    Liest eine JSON-Datei und extrahiert alle String-Werte.
    Gibt sie als zusammenhängenden Text zurück.
    """
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    strings = _extract_strings(data)
    return "\n".join(strings)


def _extract_strings(obj, depth: int = 0) -> List[str]:
    """Extrahiert rekursiv alle String-Werte aus einem JSON-Objekt."""
    strings = []
    if depth > 50:  # Schutz vor zu tiefer Rekursion
        return strings

    if isinstance(obj, str):
        if obj.strip():  # Leere Strings ignorieren
            strings.append(obj.strip())
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(_extract_strings(value, depth + 1))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(_extract_strings(item, depth + 1))
    return strings


def _read_pdf_file(path: Path) -> str:
    """
    Liest eine PDF-Datei mit pdfplumber (optionale Abhängigkeit).

    Installation: pip install easy-rag[pdf]
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber wird für PDF-Dateien benötigt. "
            "Installiere es mit: pip install easy-rag[pdf]"
        )

    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "",
) -> List[Chunk]:
    """
    Zerlegt Text in Chunks mit überlappenden Rändern.

    Versucht intelligent an Absatz- oder Satzgrenzen zu trennen,
    anstatt mitten im Wort abzuschneiden.

    Args:
        text: Der zu zerlegende Text
        chunk_size: Maximale Zeichenanzahl pro Chunk
        overlap: Überlappung zwischen benachbarten Chunks
        source: Dateipfad als Quellenangabe

    Returns:
        Liste von Chunk-Objekten
    """
    if not text or not text.strip():
        return []

    # Normalisiere Whitespace
    text = text.strip()

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        # Ende des aktuellen Chunks berechnen
        end = min(start + chunk_size, len(text))

        # Wenn wir nicht am Textende sind, suche einen guten Trennpunkt
        if end < len(text):
            end = _find_split_point(text, start, end)

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append(
                Chunk(
                    text=chunk_text_content,
                    source=source,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_idx += 1

        # Nächster Start: aktuelles Ende minus Überlappung
        start = max(start + 1, end - overlap)

        # Vermeide Endlosschleifen
        if start >= len(text):
            break

    return chunks


def _find_split_point(text: str, start: int, end: int) -> int:
    """
    Sucht den besten Trennpunkt nahe 'end'.

    Priorität:
    1. Absatzgrenze (Doppel-Newline)
    2. Zeilenumbruch
    3. Satzende (. ! ?)
    4. Wortgrenze (Leerzeichen)
    5. Falls nichts gefunden: 'end' bleibt
    """
    # Suchbereich: letztes Viertel des Chunks
    search_start = start + (end - start) * 3 // 4
    search_text = text[search_start:end]

    # 1. Absatzgrenze suchen
    para_pos = search_text.rfind("\n\n")
    if para_pos != -1:
        return search_start + para_pos + 2

    # 2. Zeilenumbruch suchen
    newline_pos = search_text.rfind("\n")
    if newline_pos != -1:
        return search_start + newline_pos + 1

    # 3. Satzende suchen
    for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        sep_pos = search_text.rfind(sep)
        if sep_pos != -1:
            return search_start + sep_pos + len(sep)

    # 4. Wortgrenze suchen
    space_pos = search_text.rfind(" ")
    if space_pos != -1:
        return search_start + space_pos + 1

    # 5. Kein guter Trennpunkt gefunden
    return end


def load_documents(
    path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Lädt alle unterstützten Dokumente aus einem Verzeichnis oder einer einzelnen Datei.

    Args:
        path: Dateipfad oder Verzeichnispfad
        chunk_size: Maximale Zeichenanzahl pro Chunk
        overlap: Überlappung zwischen Chunks

    Returns:
        Liste aller Chunks aus allen Dokumenten
    """
    path_obj = Path(path)
    all_chunks = []

    if path_obj.is_file():
        # Einzelne Datei laden
        if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = read_file(str(path_obj))
            chunks = chunk_text(text, chunk_size, overlap, source=str(path_obj))
            all_chunks.extend(chunks)
    elif path_obj.is_dir():
        # Alle unterstützten Dateien im Verzeichnis laden (rekursiv)
        for ext in SUPPORTED_EXTENSIONS:
            for filepath in sorted(path_obj.rglob(f"*{ext}")):
                try:
                    text = read_file(str(filepath))
                    chunks = chunk_text(
                        text, chunk_size, overlap, source=str(filepath)
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    # Fehlende Dateien oder Lesefehler überspringen
                    print(f"Warnung: Konnte '{filepath}' nicht laden: {e}")
    else:
        raise FileNotFoundError(f"Pfad nicht gefunden: {path}")

    return all_chunks
