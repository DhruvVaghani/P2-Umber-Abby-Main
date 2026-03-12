#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = BASE_DIR / "mock_data" / "policies"
OUTPUT_DIR = BASE_DIR / "mock_data"
CHROMA_DIR = OUTPUT_DIR / "chroma"
COLLECTION_NAME = "policy_chunks"


def chunk_markdown(text: str, size: int = 350) -> list[tuple[str, str]]:
    current_section = "overview"
    current_text = ""
    chunks: list[tuple[str, str]] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_text.strip():
                chunks.extend(split_large_chunk(current_section, current_text, size))
                current_text = ""
            current_section = line.lstrip("#").strip() or "overview"
            current_text = f"{line}\n"
            continue

        next_line = f"{line}\n"
        if len(current_text) + len(next_line) > size and current_text.strip():
            chunks.append((current_section, current_text.strip()))
            current_text = next_line
        else:
            current_text += next_line

    if current_text.strip():
        chunks.extend(split_large_chunk(current_section, current_text, size))

    return chunks


def split_large_chunk(section: str, text: str, size: int) -> list[tuple[str, str]]:
    if len(text) <= size:
        return [(section, text.strip())]

    parts: list[tuple[str, str]] = []
    words = text.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            parts.append((section, current))
            current = word
        else:
            current = candidate
    if current:
        parts.append((section, current))
    return parts


def build_index() -> dict[str, list[dict[str, str | int]]]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    #collection.delete(where={})

    docs: list[dict[str, str | int]] = []
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for policy_file in sorted(POLICY_DIR.glob("*.md")):
        text = policy_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        for index, (section, chunk_text) in enumerate(chunks, start=1):
            doc_id = f"{policy_file.name}#chunk-{index}"
            metadata = {
                "doc_id": doc_id,
                "file_name": policy_file.name,
                "section": section,
                "chunk_index": index,
            }
            docs.append(
                {
                    "doc_id": doc_id,
                    "file_name": policy_file.name,
                    "section": section,
                    "chunk_index": index,
                    "text": chunk_text,
                }
            )
            ids.append(doc_id)
            documents.append(chunk_text)
            metadatas.append(metadata)

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    output_path = OUTPUT_DIR / "policy_index.json"
    output_path.write_text(json.dumps({"docs": docs}, indent=2), encoding="utf-8")
    return {"docs": docs}


if __name__ == "__main__":
    result = build_index()
    print(f"Indexed {len(result['docs'])} chunks into Chroma collection '{COLLECTION_NAME}'.")
