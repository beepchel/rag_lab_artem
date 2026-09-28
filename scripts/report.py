import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.evaluation import load_questions

K = "3"


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|---|" + "---:|" * (len(headers) - 1))
    for row in rows:
        cells = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        print("| " + " | ".join(cells) + " |")
    print()


def mean(rows, k, name):
    return sum(row["metrics"][k][name] for row in rows) / len(rows)


def main():
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dev = {question["id"]: question
           for question in load_questions(ROOT / "data" / "dev_questions.json")}
    table(["k", "Hit@k", "Recall@k", "Precision@k", "MRR@k"],
          [[f"k={k}", m["hit"], m["recall"], m["precision"], m["reciprocal_rank"]]
           for k, m in report["aggregate"].items()])
    for field in ("category", "difficulty"):
        groups = {}
        for row in report["questions"]:
            groups.setdefault(dev[row["id"]][field], []).append(row)
        table([field, "n", "Hit@1", f"Hit@{K}", f"MRR@{K}"],
              [[name, len(rows), mean(rows, "1", "hit"), mean(rows, K, "hit"),
                mean(rows, K, "reciprocal_rank")] for name, rows in sorted(groups.items())])


if __name__ == "__main__":
    main()
