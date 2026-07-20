"""
test_rag.py — quick standalone test of vector_search(), no Plivo/Gemini Live call needed.

Usage:
    python test_rag.py "your test query here"
"""

import asyncio
import sys
from rag_service import vector_search


async def main():
    query = " ".join(sys.argv[1:]) or "what does IEI do"
    print(f"\nQuery: {query}\n{'-'*50}")
    result = await vector_search(query)
    print(result)
    print(f"\n{'-'*50}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())