import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder
from rag_lab.evaluation import evaluate, load_questions
from rag_lab.retrieval import Retriever

DEV = ROOT / "data" / "dev_questions.json"
CHUNK_SIZE = 120
KS = [1, 3, 5, 10]
HEADERS = ["Конфигурация", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Recall@5", "MRR@5",
           "Число фрагментов", "Средняя длина"]
COLUMNS = [("1", "hit"), ("3", "hit"), ("5", "hit"), ("10", "hit"), ("5", "recall"),
           ("5", "reciprocal_rank")]


def run(overlap, documents, questions):
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=overlap)
    retriever = Retriever(chunks, TfidfEmbedder())
    runs = {k: evaluate(retriever, questions, ks=[k]) for k in KS}
    report = runs[KS[-1]]
    report["aggregate"] = {str(k): runs[k]["aggregate"][str(k)] for k in KS}
    for i, row in enumerate(report["questions"]):
        row["metrics"] = {str(k): runs[k]["questions"][i]["metrics"][str(k)] for k in KS}
    report["configuration"] = {
        "backend": "tfidf",
        "chunk_size": CHUNK_SIZE,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "avg_length": round(sum(chunk.token_count for chunk in chunks) / len(chunks), 2),
        "questions": "data/dev_questions.json",
        "questions_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
    }
    return report


def main():
    output_dir = Path(sys.argv[1])
    overlaps = [int(value) for value in sys.argv[2:]]
    documents = load_documents(ROOT / "documents")
    questions = load_questions(DEV)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("| " + " | ".join(HEADERS) + " |")
    print("|---|" + "---:|" * (len(HEADERS) - 1))
    rows = []
    for overlap in overlaps:
        report = run(overlap, documents, questions)
        path = output_dir / f"overlap_{overlap}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
                        newline="\n")
        config, aggregate = report["configuration"], report["aggregate"]
        rows += [{"backend": "tfidf", "chunk_size": CHUNK_SIZE, "overlap": overlap, "k": k,
                  **metrics} for k, metrics in aggregate.items()]
        label = f"overlap={overlap}" + (" (baseline)" if overlap == 20 else "")
        cells = [f"{aggregate[k][name]:.3f}" for k, name in COLUMNS]
        print(f"| {label} | {' | '.join(cells)} | {config['chunk_count']} | "
              f"{config['avg_length']:.2f} |")
    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
