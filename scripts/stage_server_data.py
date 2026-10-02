"""
Stage raw KB files in the production server layout.

Creates:
  build/server_data_stage/arxiv_index.db
  build/server_data_stage/papers_meta.json
  build/server_data_stage/papers/quant/<topic>/.topic.json
  build/server_data_stage/papers/quant/<topic>/<pdf filename>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).parent.parent
DEFAULT_RESEARCH_ROOT = Path(r"C:\Users\ansar\Documents\Research Knowledge Base\research")
DEFAULT_STAGE_ROOT = ROOT / "build" / "server_data_stage"


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        dst.hardlink_to(src)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    args = parser.parse_args()

    research_root = args.research_root
    stage_root = args.stage_root
    papers_meta_path = research_root / "papers_meta.json"
    arxiv_db_path = research_root / "arxiv_index.db"

    with open(ROOT / "domains.yaml", encoding="utf-8") as f:
        domains = yaml.safe_load(f)["domains"]
    with open(papers_meta_path, encoding="utf-8") as f:
        papers_meta = json.load(f)

    quant_root = stage_root / "papers" / "quant"
    quant_topics = domains["quant"]["topics"]
    for topic_key, cfg in quant_topics.items():
        topic_dir = quant_root / topic_key
        topic_dir.mkdir(parents=True, exist_ok=True)
        (topic_dir / ".topic.json").write_text(
            json.dumps(
                {
                    "key": topic_key,
                    "label": cfg["label"],
                    "parent": cfg.get("parent"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    stats = {"linked": 0, "copied": 0, "exists": 0, "missing": 0}
    missing: list[str] = []
    for source_id, meta in papers_meta.items():
        topic = meta.get("topic_group") or "quant_finance"
        src = Path(meta.get("local_path") or "")
        if not src.exists():
            stats["missing"] += 1
            missing.append(source_id)
            continue
        dst = quant_root / topic / src.name
        result = link_or_copy(src, dst)
        stats[result] += 1

    link_or_copy(papers_meta_path, stage_root / "papers_meta.json")
    if arxiv_db_path.exists():
        link_or_copy(arxiv_db_path, stage_root / "arxiv_index.db")
    else:
        print(f"WARN: {arxiv_db_path} not found")

    print(json.dumps({"stage_root": str(stage_root), "stats": stats, "missing": missing[:25]}, indent=2))


if __name__ == "__main__":
    main()
