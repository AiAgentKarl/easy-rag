"""
CLI für easy-rag — Dokumente durchsuchen von der Kommandozeile.

Nutzung:
    easy-rag ./docs/ "Meine Frage"
    easy-rag ./docs/ --stats
    easy-rag ./vertrag.pdf "Was steht über Kündigung?"
"""

from __future__ import annotations

import argparse
import sys

from easy_rag.core import EasyRAG


def main():
    """Haupteinstiegspunkt für die CLI."""
    parser = argparse.ArgumentParser(
        prog="easy-rag",
        description="RAG in 3 Zeilen Code — Dokumente durchsuchen",
    )
    parser.add_argument(
        "path",
        help="Verzeichnis oder Datei zum Durchsuchen",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Suchanfrage (optional wenn --stats genutzt wird)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Anzahl der Ergebnisse (Standard: 3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk-Größe in Zeichen (Standard: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Überlappung zwischen Chunks (Standard: 50)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Zeige Index-Statistiken",
    )

    args = parser.parse_args()

    # Index erstellen
    print(f"Lade Dokumente aus: {args.path}")
    try:
        rag = EasyRAG(args.path, chunk_size=args.chunk_size, overlap=args.overlap)
    except FileNotFoundError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(1)

    # Statistiken anzeigen
    if args.stats:
        stats = rag.stats()
        print(f"\nIndex-Statistiken:")
        print(f"  Chunks:         {stats['num_chunks']}")
        print(f"  Quellen:        {stats['num_sources']}")
        print(f"  Unique Terms:   {stats['num_unique_terms']}")
        print(f"  Chunk-Größe:    {stats['chunk_size']}")
        print(f"  Überlappung:    {stats['overlap']}")
        print(f"\nQuellen:")
        for source, count in stats["sources"].items():
            print(f"    {source}: {count} Chunks")
        if not args.query:
            return

    # Suche durchführen
    if not args.query:
        parser.print_help()
        print("\nFehler: Suchanfrage oder --stats erforderlich.", file=sys.stderr)
        sys.exit(1)

    print(f"Suche nach: {args.query}\n")
    result = rag.ask(args.query, top_k=args.top_k)
    print(result.answer)


if __name__ == "__main__":
    main()
