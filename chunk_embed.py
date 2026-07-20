"""
IEI PDF → regex cleanup → LangChain hierarchical splitter → MongoDB insert
Usage: python iei_chunk_ingest.py
"""

import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
PDF_PATH       = "ieitheory.pdf"
MONGO_URI      = "mongodb+srv://hardik:hardik2005@cluster0.1ijo6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME        = "iei_voice_agent"
COLLECTION     = "iei_knowledge_chunks"

# Chunking params
CHUNK_SIZE     = 600    # chars per chunk
CHUNK_OVERLAP  = 100    # overlap between consecutive chunks


# ─────────────────────────────────────────────
# STEP 1 — Extract raw text from PDF
# ─────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)


# ─────────────────────────────────────────────
# STEP 2 — Regex cleanup
# This PDF has no spaces between words (font encoding issue).
# Pipeline:
#   1. Fix missing spaces before uppercase runs / markdown tokens
#   2. Strip Python wrapper (IEI_KNOWLEDGE = """ ... """)
#   3. Normalize whitespace, dashes, bullets
# ─────────────────────────────────────────────
def regex_cleanup(raw: str) -> str:
    text = raw

    # Fix squished words: insert space before uppercase after lowercase
    # e.g. "engineeringdivisions" → NOT fixable purely by regex,
    # but markdown headers and field labels ARE recoverable:
    text = re.sub(r'(\w)(#{1,4})', r'\1\n\2', text)          # header stuck to prev word
    text = re.sub(r'(#{1,4})(\w)', r'\1 \2', text)           # header has no space after #
    text = re.sub(r'(\*{1,2})(\w)', r'\1 \2', text)          # bold marker stuck to word
    text = re.sub(r'(\w)(\*{1,2})', r'\1 \2', text)

    # Insert spaces between lowercase-then-uppercase transitions
    # (handles squished words from font stripping)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Insert space after colon if missing before capital
    text = re.sub(r':([A-Z])', r': \1', text)

    # Insert space after period if missing
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)

    # Strip Python file wrapper lines
    text = re.sub(r'^#.*?\.py\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'IEI_KNOWLEDGE\s*=\s*"""', '', text)
    text = re.sub(r'"""\s*$', '', text)

    # Normalize markdown separators
    text = re.sub(r'-{3,}', '\n---\n', text)

    # Normalize bullet points
    text = re.sub(r'^\s*[-•]\s*', '- ', text, flags=re.MULTILINE)

    # Collapse excessive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return text


# ─────────────────────────────────────────────
# STEP 3 — LangChain Hierarchical (Recursive) Splitter
# Splits on: section headers → paragraphs → sentences → chars
# ─────────────────────────────────────────────
def chunk_text(text: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n## ",    # H2 section  (highest priority)
            "\n### ",   # H3 subsection
            "\n#### ",  # H4
            "\n---\n",  # horizontal rule / separator
            "\n\n",     # paragraph break
            "\n",       # line break
            ". ",       # sentence
            " ",        # word (last resort)
        ],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text)

    # Build structured chunk docs
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < 30:          # skip noise / near-empty chunks
            continue

        # Detect section heading from first non-empty line
        first_line = next((l for l in chunk.splitlines() if l.strip()), "")
        heading = re.sub(r'^#+\s*', '', first_line).strip()[:120]

        chunks.append({
            "chunk_index": i,
            "heading":     heading,
            "content":     chunk,
            "char_count":  len(chunk),
            "source":      "ieitheory.pdf",
            "doc_type":    "iei_knowledge_base",
        })

    return chunks


# ─────────────────────────────────────────────
# STEP 4 — Insert into MongoDB
# ─────────────────────────────────────────────
def insert_to_mongo(chunks: list[dict]):
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    col    = db[COLLECTION]

    # Optional: clear existing chunks for this source before reinserting
    deleted = col.delete_many({"source": "ieitheory.pdf"})
    print(f"[MongoDB] Cleared {deleted.deleted_count} existing chunks")

    # Stamp each doc with ingestion time
    now = datetime.now(timezone.utc)
    for chunk in chunks:
        chunk["ingested_at"] = now

    result = col.insert_many(chunks)
    print(f"[MongoDB] Inserted {len(result.inserted_ids)} chunks into {DB_NAME}.{COLLECTION}")

    # Create text index for keyword search (runs once, no-op if exists)
    col.create_index([("content", "text"), ("heading", "text")], name="content_text_idx")
    print("[MongoDB] Text index ensured on content + heading")

    client.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[1/4] Extracting text from {PDF_PATH}...")
    raw = extract_text(PDF_PATH)
    print(f"      Raw chars: {len(raw)}")

    print("[2/4] Running regex cleanup...")
    clean = regex_cleanup(raw)
    print(f"      Clean chars: {len(clean)}")

    # Optional: inspect cleaned text
    # with open("iei_clean_debug.txt", "w") as f:
    #     f.write(clean)

    print("[3/4] Chunking with LangChain RecursiveCharacterTextSplitter...")
    chunks = chunk_text(clean)
    print(f"      Total chunks: {len(chunks)}")
    print(f"      Avg chunk size: {sum(c['char_count'] for c in chunks)//len(chunks)} chars")

    # Preview first 2 chunks
    for c in chunks[:2]:
        print(f"\n  --- Chunk {c['chunk_index']} | heading: {c['heading'][:60]}")
        print(f"  {c['content'][:200]}")

    print("\n[4/4] Inserting into MongoDB...")
    insert_to_mongo(chunks)

    print("\n✓ Done.")