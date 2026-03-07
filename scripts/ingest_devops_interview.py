#!/usr/bin/env python3
"""Ingest DevOps interview questions into the RAG system.

Reads datasets from the devops-interview-questions repo and appends
document + QA entries to the llm_interview_note RAG JSONL files.

Sources:
  - devops_rag_answered_full_405.json  (405 zh records, 24 categories)
  - devops_quiz_10_en.json             (10 en DevOps beginner)
  - jenkins_beginner_24_33_en.json     (10 en Jenkins beginner)
  - jenkins_advanced_34_40_en.json     (7  en Jenkins advanced)
"""

import json
import os
from datetime import datetime, timezone

DEVOPS_DATA = "/home/calelin/dev/devops-interview-questions/data"
RAG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DOCS_FILE = os.path.join(RAG_DIR, "all_documents.jsonl")
QA_FILE = os.path.join(RAG_DIR, "all_qa_pairs.jsonl")
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
SOURCE_REPO = "https://github.com/aliaskov/devops-interview-questions"

# Map devops-interview-questions categories → llm_interview_note categories
CATEGORY_MAP = {
    "devops": ("11.DevOps", "DevOps基础"),
    "jenkins": ("11.DevOps", "Jenkins"),
    "docker": ("11.DevOps", "Docker"),
    "kubernetes": ("11.DevOps", "Kubernetes"),
    "terraform": ("11.DevOps", "Terraform"),
    "ansible": ("11.DevOps", "Ansible"),
    "prometheus": ("11.DevOps", "Prometheus"),
    "aws": ("11.DevOps", "AWS"),
    "azure": ("11.DevOps", "Azure"),
    "gcp": ("11.DevOps", "GCP"),
    "cloud": ("11.DevOps", "Cloud"),
    "linux": ("11.DevOps", "Linux"),
    "network": ("11.DevOps", "Networking"),
    "git": ("11.DevOps", "Git"),
    "python": ("11.DevOps", "Python"),
    "go": ("11.DevOps", "Go"),
    "shell": ("11.DevOps", "Shell"),
    "sql": ("11.DevOps", "SQL"),
    "mongo": ("11.DevOps", "MongoDB"),
    "security": ("11.DevOps", "Security"),
    "puppet": ("11.DevOps", "Puppet"),
    "openstack": ("11.DevOps", "OpenStack"),
    "openshift": ("11.DevOps", "OpenShift"),
    "coding": ("11.DevOps", "Coding"),
}


def load_json(filename):
    path = os.path.join(DEVOPS_DATA, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_zh(rec):
    """Normalize a zh-CN 405 record into a common dict."""
    return {
        "orig_id": rec["id"],
        "category": rec.get("category", "devops"),
        "difficulty": rec.get("difficulty", "intermediate"),
        "question": rec.get("question_zh", ""),
        "keywords": rec.get("keywords", []),
        "short_answer": rec.get("short_answer_zh", ""),
        "detailed_answer": rec.get("detailed_answer_zh", ""),
        "key_points": rec.get("key_points_zh", []),
        "example": rec.get("example_zh", ""),
        "related_topics": rec.get("related_topics", []),
        "language": "zh-CN",
    }


def normalize_en_quiz(rec):
    """Normalize an English quiz-10 record."""
    return {
        "orig_id": rec["id"],
        "category": rec.get("category", "devops"),
        "difficulty": rec.get("difficulty", "beginner"),
        "question": rec.get("question_en", ""),
        "keywords": rec.get("keywords", []),
        "short_answer": rec.get("short_answer_en", ""),
        "detailed_answer": rec.get("detailed_answer_en", ""),
        "key_points": rec.get("key_points_en", []),
        "example": rec.get("example_en", ""),
        "related_topics": rec.get("related_topics", []),
        "language": "en",
    }


def normalize_en_jenkins(rec):
    """Normalize an English Jenkins record."""
    return {
        "orig_id": rec["id"],
        "category": rec.get("category", "jenkins"),
        "difficulty": rec.get("difficulty", "beginner"),
        "question": rec.get("question", ""),
        "keywords": rec.get("keywords", []),
        "short_answer": rec.get("short_answer", ""),
        "detailed_answer": rec.get("detailed_answer", ""),
        "key_points": rec.get("key_points", []),
        "example": rec.get("example", ""),
        "related_topics": rec.get("related_topics", []),
        "language": "en",
    }


def make_document(idx, rec):
    cat, subcat = CATEGORY_MAP.get(rec["category"], ("11.DevOps", rec["category"]))
    content_parts = [f"# {rec['question']}", ""]
    if rec["short_answer"]:
        content_parts.append(rec["short_answer"])
        content_parts.append("")
    if rec["detailed_answer"]:
        content_parts.append(rec["detailed_answer"])
        content_parts.append("")
    if rec["example"]:
        content_parts.append(f"Example: {rec['example']}")
    content = "\n".join(content_parts).strip()

    return {
        "id": f"doc_devops_{idx:04d}",
        "category": cat,
        "subcategory": subcat,
        "title": rec["question"][:120],
        "content": content,
        "questions": [rec["question"]],
        "keywords": rec["keywords"],
        "difficulty": rec["difficulty"],
        "source_file": f"devops-interview-questions/{rec['orig_id']}",
        "url": SOURCE_REPO,
        "last_updated": NOW,
        "metadata": {
            "word_count": len(content.split()),
            "has_code": "```" in content or "sh " in content,
            "has_images": False,
            "references": [SOURCE_REPO],
        },
    }


def make_qa(idx, rec):
    cat, subcat = CATEGORY_MAP.get(rec["category"], ("11.DevOps", rec["category"]))
    return {
        "id": f"qa_devops_{idx:04d}",
        "category": cat,
        "subcategory": subcat,
        "difficulty": rec["difficulty"],
        "question": rec["question"],
        "short_answer": rec["short_answer"],
        "detailed_answer": rec["detailed_answer"],
        "key_points": rec["key_points"] if isinstance(rec["key_points"], list) else [],
        "code_examples": [rec["example"]] if rec["example"] else [],
        "related_topics": rec["related_topics"] if isinstance(rec["related_topics"], list) else [],
        "keywords": rec["keywords"],
        "source_file": f"devops-interview-questions/{rec['orig_id']}",
        "url": SOURCE_REPO,
        "status": "verified",
    }


def main():
    records = []

    # 1. Load 405 zh records
    zh_data = load_json("devops_rag_answered_full_405.json")
    for r in zh_data:
        records.append(normalize_zh(r))
    print(f"Loaded 405 zh records ({len(zh_data)})")

    # 2. Load English datasets (deduplicate by orig_id against zh set)
    seen_ids = {r["orig_id"] for r in records}

    en_quiz = load_json("devops_quiz_10_en.json")
    added_en = 0
    for r in en_quiz:
        nr = normalize_en_quiz(r)
        if nr["orig_id"] not in seen_ids:
            records.append(nr)
            seen_ids.add(nr["orig_id"])
            added_en += 1
        else:
            # Merge English answer into existing zh record
            for rec in records:
                if rec["orig_id"] == nr["orig_id"]:
                    rec["question"] = f"{rec['question']}\n\n(EN) {nr['question']}"
                    if nr["detailed_answer"]:
                        rec["detailed_answer"] += f"\n\n--- English ---\n{nr['detailed_answer']}"
                    if nr["short_answer"]:
                        rec["short_answer"] += f" | (EN) {nr['short_answer']}"
                    break
    print(f"English quiz: {len(en_quiz)} records ({added_en} new, {len(en_quiz)-added_en} merged)")

    for fname, normalizer in [
        ("jenkins_beginner_24_33_en.json", normalize_en_jenkins),
        ("jenkins_advanced_34_40_en.json", normalize_en_jenkins),
    ]:
        data = load_json(fname)
        added = 0
        for r in data:
            nr = normalizer(r)
            if nr["orig_id"] not in seen_ids:
                records.append(nr)
                seen_ids.add(nr["orig_id"])
                added += 1
            else:
                for rec in records:
                    if rec["orig_id"] == nr["orig_id"]:
                        if nr["detailed_answer"]:
                            rec["detailed_answer"] += f"\n\n--- English ---\n{nr['detailed_answer']}"
                        break
        print(f"{fname}: {len(data)} records ({added} new, {len(data)-added} merged)")

    print(f"\nTotal records to ingest: {len(records)}")

    # Build documents and QA pairs
    docs = []
    qas = []
    for idx, rec in enumerate(records, start=1):
        docs.append(make_document(idx, rec))
        qas.append(make_qa(idx, rec))

    # Append to existing JSONL files
    with open(DOCS_FILE, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(QA_FILE, "a", encoding="utf-8") as f:
        for qa in qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\nAppended {len(docs)} documents to {DOCS_FILE}")
    print(f"Appended {len(qas)} QA pairs to {QA_FILE}")


if __name__ == "__main__":
    main()
