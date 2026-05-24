import os
import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

base_path: Path = Path(os.getenv("BASE_PATH", "/tmp/knowledgebase"))

DOCS_DIR = base_path / "docs"
INDEX_PATH = base_path / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    parsed_sections = []
    heading_stack = []
    current_heading = None
    current_content_lines = []

    def emit_section():
        if current_heading is not None:
            content = "\n".join(current_content_lines).strip()
            heading_path = list(heading_stack)
            token_text = " ".join(heading_path) + "\n" + content
            tokens = tokenize(token_text)
            
            sect = Section(
                id=f"{path.name}#{slugify(current_heading)}",
                file=path.name,
                heading=current_heading,
                heading_path=heading_path,
                content=content,
                tokens=tokens
            )
            parsed_sections.append(sect)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = HEADING_RE.match(line)
            if match:
                emit_section()
                hashes, title = match.groups()
                level = len(hashes)
                title = title.strip()
                
                if len(heading_stack) >= level:
                    heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)
                
                current_heading = title
                current_content_lines = []
            else:
                if current_heading is not None:
                    current_content_lines.append(line.rstrip("\n"))
                    
    emit_section()
    return parsed_sections


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [sect.to_dict() for sect in sections],
        "stats": {
            "files_indexed": files_indexed,
            "avg_doc_len": avg_doc_len,
            "doc_freq": dict(doc_freq),
        }
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rebuild_stats() -> None:
    global sections, doc_freq, avg_doc_len, files_indexed
    
    unique_files = {sect.file for sect in sections}
    files_indexed = len(unique_files)
    
    doc_freq.clear()
    total_tokens = 0
    for sect in sections:
        unique_tokens = set(sect.tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
        total_tokens += len(sect.tokens)
        
    if sections:
        avg_doc_len = total_tokens / len(sections)
    else:
        avg_doc_len = 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    global sections
    if not index_path.exists():
        return 0, 0
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sections = []
    for item in payload.get("sections", []):
        sect = Section(
            id=item["id"],
            file=item["file"],
            heading=item["heading"],
            heading_path=item["heading_path"],
            content=item["content"],
            tokens=item["tokens"],
        )
        sections.append(sect)
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections
    sections = []
    if docs_dir.exists():
        for path in sorted(docs_dir.glob("*.md")):
            sections.extend(parse_markdown(path))
    rebuild_stats()
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    global sections, doc_freq, avg_doc_len
    
    score = 0.0
    N = len(sections)
    if N == 0:
        return 0.0
        
    doc_len = len(section.tokens)
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    
    for q in set(query_tokens):
        tf = section.tokens.count(q)
        nq = doc_freq.get(q, 0)
        
        idf = math.log(1.0 + (N - nq + 0.5) / (nq + 0.5))
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len)) if avg_doc_len > 0 else tf + k1
        
        term_score = idf * (numerator / denominator) if denominator > 0 else 0.0
        
        if q in heading_tokens:
            term_score *= 1.2
            
        score += term_score
        
    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
