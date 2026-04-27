from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}

SECRET_PATTERNS = [
    re.compile(r"\.env$", re.IGNORECASE),
    re.compile(r"\.token$", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
]

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "jest",
    "oraz",
    "oraz",
    "jak",
    "nie",
    "sie",
    "się",
    "dla",
    "you",
    "your",
    "image",
    "model",
    "prompt",
    "claude",
    "grok",
}


@dataclass
class SourceCard:
    source: str
    title: str
    kind: str
    sha256: str
    chars: int
    excerpt: str
    tags: list[str]


@dataclass
class BrainSnapshot:
    snapshot_id: str
    created_at: str
    source_count: int
    source_cards: list[SourceCard]
    top_terms: list[str]
    operating_memory: list[str]
    next_actions: list[str]


def should_skip_path(path: str) -> bool:
    parts = re.split(r"[\\/]+", path)
    if any(part in SKIP_PARTS for part in parts):
        return True
    name = parts[-1] if parts else path
    return any(pattern.search(name) for pattern in SECRET_PATTERNS)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def decode_bytes(raw: bytes) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def iter_text_files(path: Path, max_file_bytes: int) -> Iterable[tuple[str, str]]:
    if path.is_dir():
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            rel = str(item.relative_to(path))
            if should_skip_path(rel) or item.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if item.stat().st_size > max_file_bytes:
                continue
            text = decode_bytes(item.read_bytes())
            if text:
                yield f"{path.name}/{rel}", text
        return

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if should_skip_path(info.filename):
                    continue
                if Path(info.filename).suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if info.file_size > max_file_bytes:
                    continue
                text = decode_bytes(archive.read(info))
                if text:
                    yield f"{path.name}:{info.filename}", text
        return

    if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not should_skip_path(path.name):
        if path.stat().st_size <= max_file_bytes:
            text = decode_bytes(path.read_bytes())
            if text:
                yield str(path), text


def tags_for(text: str) -> list[str]:
    lowered = text.lower()
    candidates = {
        "memory": ["memory", "pamię", "pamiec", "snapshot", "context"],
        "prompting": ["prompt", "structured", "xml", "schema"],
        "vision": ["image", "photo", "camera", "comfy", "diffusion", "gamma"],
        "automation": ["mcp", "plugin", "workflow", "agent", "tool"],
        "security": ["security", "token", "secret", "watchdog", "cerber"],
        "coding": ["python", "typescript", "github", "test", "repo"],
    }
    tags = [tag for tag, needles in candidates.items() if any(needle in lowered for needle in needles)]
    return tags[:5] or ["general"]


def title_for(source: str, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" #\t\r\n")
        if 6 <= len(stripped) <= 100:
            return stripped
    return Path(source).name[:100]


def source_card(source: str, text: str, max_excerpt_chars: int) -> SourceCard:
    cleaned = clean_text(text)
    excerpt = cleaned[:max_excerpt_chars]
    digest = hashlib.sha256(cleaned.encode("utf-8", errors="ignore")).hexdigest()
    return SourceCard(
        source=source,
        title=title_for(source, text),
        kind=Path(source.split(":")[-1]).suffix.lower().lstrip(".") or "text",
        sha256=digest,
        chars=len(cleaned),
        excerpt=excerpt,
        tags=tags_for(cleaned),
    )


def top_terms(cards: list[SourceCard], limit: int = 20) -> list[str]:
    words: Counter[str] = Counter()
    for card in cards:
        text = f"{card.title} {card.excerpt}".lower()
        for word in re.findall(r"[a-ząćęłńóśźż0-9_-]{4,}", text):
            if word not in STOP_WORDS and not word.isdigit():
                words[word] += 1
    return [word for word, _ in words.most_common(limit)]


def build_operating_memory(cards: list[SourceCard], terms: list[str]) -> list[str]:
    tag_counts = Counter(tag for card in cards for tag in card.tags)
    strongest_tags = ", ".join(tag for tag, _ in tag_counts.most_common(6))
    return [
        "ALFA Brain keeps continuity by turning long source material into compact snapshots.",
        f"Current source map emphasizes: {strongest_tags}.",
        f"Useful recurring terms: {', '.join(terms[:10])}.",
        "Use snapshots as context, not as unquestioned truth; verify live repo state before edits.",
        "Keep secrets out of snapshots; source ingestion skips common token and environment files.",
    ]


def build_snapshot(paths: list[Path], max_file_bytes: int, max_excerpt_chars: int) -> BrainSnapshot:
    cards: list[SourceCard] = []
    seen: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        for source, text in iter_text_files(path, max_file_bytes=max_file_bytes):
            card = source_card(source, text, max_excerpt_chars=max_excerpt_chars)
            if card.sha256 in seen:
                continue
            seen.add(card.sha256)
            cards.append(card)

    terms = top_terms(cards)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    snapshot_id = hashlib.sha256(f"{now}:{len(cards)}:{','.join(terms[:8])}".encode()).hexdigest()[:12]

    return BrainSnapshot(
        snapshot_id=snapshot_id,
        created_at=now,
        source_count=len(cards),
        source_cards=cards,
        top_terms=terms,
        operating_memory=build_operating_memory(cards, terms),
        next_actions=[
            "Inject the latest cumulative memory at the start of the next AI session.",
            "Create a fresh snapshot before the conversation exceeds the context budget.",
            "Promote stable decisions into project instructions; leave temporary observations in snapshots.",
        ],
    )


def write_snapshot(snapshot: BrainSnapshot, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snapshot-{snapshot.snapshot_id}.json"
    path.write_text(json.dumps(asdict(snapshot), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_snapshots(memory_dir: Path) -> list[dict]:
    snapshots = []
    for path in sorted(memory_dir.glob("snapshot-*.json")):
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(snapshots, key=lambda snapshot: snapshot.get("created_at", ""))


def render_injection(snapshots: list[dict], max_cards: int = 24) -> str:
    latest = snapshots[-1] if snapshots else None
    cards = []
    for snapshot in snapshots:
        cards.extend(snapshot.get("source_cards", []))
    cards = cards[-max_cards:]

    lines = [
        "<alfa_brain>",
        "Purpose: preserve continuity across AI conversations by injecting compact project memory.",
    ]
    if latest:
        lines.extend(
            [
                f"Latest snapshot: {latest['snapshot_id']} at {latest['created_at']}",
                f"Total snapshots: {len(snapshots)}",
                "",
                "<operating_memory>",
            ]
        )
        lines.extend(f"- {item}" for item in latest.get("operating_memory", []))
        lines.append("</operating_memory>")
        lines.append("")
        lines.append("<top_terms>")
        lines.append(", ".join(latest.get("top_terms", [])[:20]))
        lines.append("</top_terms>")

    lines.append("")
    lines.append("<recent_source_cards>")
    for card in cards:
        tags = ", ".join(card.get("tags", []))
        lines.append(f"- {card.get('title')} [{tags}]")
        lines.append(f"  Source: {card.get('source')}")
        lines.append(f"  Note: {card.get('excerpt', '')[:280]}")
    lines.append("</recent_source_cards>")
    lines.append("")
    lines.append("<session_rules>")
    lines.append("- Start by reading this memory, then inspect live files before making changes.")
    lines.append("- When context is getting full, create a new snapshot and summarize decisions.")
    lines.append("- Keep API keys, tokens, private credentials, and raw personal data out of memory files.")
    lines.append("</session_rules>")
    lines.append("</alfa_brain>")
    return "\n".join(lines)


def load_source_config(path: Path) -> list[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Path(item).expanduser() for item in data["sources"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ALFA Brain context snapshots and injections.")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="Create a snapshot from local folders, files, and zip archives.")
    snap.add_argument("--sources", type=Path, required=True)
    snap.add_argument("--out", type=Path, default=Path("memory"))
    snap.add_argument("--max-file-bytes", type=int, default=250_000)
    snap.add_argument("--max-excerpt-chars", type=int, default=1_400)

    inj = sub.add_parser("inject", help="Render cumulative context injection from snapshots.")
    inj.add_argument("--memory", type=Path, default=Path("memory"))
    inj.add_argument("--out", type=Path, default=Path("memory/context-injection.md"))

    args = parser.parse_args()

    if args.command == "snapshot":
        sources = load_source_config(args.sources)
        snapshot = build_snapshot(sources, args.max_file_bytes, args.max_excerpt_chars)
        path = write_snapshot(snapshot, args.out)
        print(f"Wrote {path} with {snapshot.source_count} source cards")
    elif args.command == "inject":
        snapshots = load_snapshots(args.memory)
        injection = render_injection(snapshots)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(injection, encoding="utf-8")
        print(f"Wrote {args.out} from {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
