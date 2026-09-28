import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder, _tokens
from rag_lab.retrieval import Retriever

CHUNK_SIZE = 120


def main():
    query, gold, overlap, output = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    chunks = chunk_documents(load_documents(ROOT / "documents"), chunk_size=CHUNK_SIZE,
                             overlap=overlap)
    embedder = TfidfEmbedder()
    results = Retriever(chunks, embedder).retrieve(query, top_k=len(chunks))
    words = embedder._vocabulary

    tokens = _tokens(query)
    lines = [f"query: {query}",
             f"gold: {gold}; tfidf, chunk_size {CHUNK_SIZE}, overlap {overlap}",
             "not in corpus vocabulary: " + (", ".join(sorted(set(tokens) - set(words))) or "—"),
             "idf: " + (", ".join(f"{token}={embedder._idf[words[token]]:.2f}"
                                  for token in dict.fromkeys(tokens) if token in words) or "—")]

    shown = results[:3]
    if gold not in {result.chunk.document_id for result in shown}:
        shown.append(next(result for result in results if result.chunk.document_id == gold))
    query_vector = embedder.embed_many([query])[0]
    for result in shown:
        chunk_vector = embedder.embed_many([result.chunk.text])[0]
        parts = sorted(((query_vector[i] * chunk_vector[i], word) for word, i in words.items()
                        if query_vector[i] * chunk_vector[i]), reverse=True)
        lines += ["", f"#{result.rank} {result.chunk.document_id} words={result.chunk.token_count} "
                      f"cosine={result.score:.4f}",
                  "   " + ", ".join(f"{word} {value:.4f}" for value, word in parts)]

    text = "\n".join(lines) + "\n"
    print(text, end="")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
