"""
Report service: generates a markdown report describing the dataset.
"""
import json
import pandas as pd
from collections import Counter
from typing import Any
import chess

from . import storage
from .. import config
from ..ml import encoding

def generate_report(username: str, samples: list[dict[str, Any]], info: dict) -> None:
    """
    Generate dataset_report.md
    """
    if not samples:
        report = "# Dataset Report\n\nNo samples extracted.\n"
        out_path = storage.player_dir(username) / "dataset_report.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        return

    df = pd.DataFrame(samples)
    
    total_positions = len(df)
    n_games = df["game_id"].nunique()
    avg_moves = total_positions / n_games if n_games > 0 else 0
    
    split_counts = df["split"].value_counts().to_dict()
    color_counts = df["color"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()
    
    # Fraction with clock
    has_clock = df["clock_before_move"].notna().sum()
    clock_frac = has_clock / total_positions if total_positions > 0 else 0
    
    # Piece type distribution
    # We need to re-parse UCI to get piece type, or we can just parse the board.
    # Actually, we can just instantiate a Board for each fen, but that's slow.
    # A faster way: string manipulation on FEN to get piece at from_square.
    # Or just use python-chess. Let's do it efficiently or just accept the few seconds.
    piece_counts = Counter()
    first_moves_w = Counter()
    replies_b = Counter()
    
    for row in samples:
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["uci"])
        piece = board.piece_at(move.from_square)
        if piece:
            piece_counts[piece.symbol().upper()] += 1
            
        # Top first moves as White (ply == 0)
        if row["ply"] == 0 and row["color"] == "white":
            first_moves_w[row["uci"]] += 1
            
        # Top replies as Black (ply == 1 and Black moving)
        if row["ply"] == 1 and row["color"] == "black":
            replies_b[row["uci"]] += 1

    lines = [
        f"# Dataset Report for {username}",
        "",
        f"**Model Tier**: {info.get('model_tier', 'unknown')}",
        "",
        "## Overall Stats",
        f"- Total Games: {n_games}",
        f"- Total Positions: {total_positions}",
        f"- Average Target Moves per Game: {avg_moves:.1f}",
        f"- Positions with Clock Data: {clock_frac:.1%}",
        "",
        "## Splits",
    ]
    
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        lines.append(f"- **{split}**: {count} positions")
        
    lines.extend([
        "",
        "## Distributions",
        "### Color",
    ])
    for c, count in color_counts.items():
        lines.append(f"- {c}: {count}")
        
    lines.append("### Speed Category")
    for cat, count in category_counts.items():
        lines.append(f"- {cat}: {count}")
        
    lines.append("### Moved Piece Type")
    for pt, count in piece_counts.most_common():
        lines.append(f"- {pt}: {count}")
        
    lines.extend([
        "",
        "## Top Openings",
        "### Top 10 First Moves (as White)"
    ])
    for uci, count in first_moves_w.most_common(10):
        lines.append(f"- {uci}: {count}")
        
    lines.append("### Top 10 Replies (as Black)")
    for uci, count in replies_b.most_common(10):
        lines.append(f"- {uci}: {count}")
        
    # Ingestion Stats
    pdir = storage.player_dir(username)
    ingest_info_path = pdir / "ingest_info.json"
    ingest_info = {}
    if ingest_info_path.exists():
        try:
            with open(ingest_info_path, "r", encoding="utf-8") as f:
                ingest_info = json.load(f)
        except Exception:
            pass

    retries_used = info.get("retries_used", ingest_info.get("retries_used", 0))
    requests = ingest_info.get("requests", 0)
    skipped_duplicates = ingest_info.get("skipped_duplicates", 0)
    bucket_requests = ingest_info.get("bucket_requests", {})
    request_log = ingest_info.get("request_log", [])
    ingest_drop_stats = ingest_info.get("drop_stats", {})

    lines.extend([
        "",
        "## Ingestion",
        f"- Retries Used: {retries_used}",
        f"- Requests: {requests}",
        f"- Skipped Duplicates: {skipped_duplicates}",
        "",
        "### Per-Bucket Requests",
    ])
    for b in ["last_6m", "6_18m", "18_36m"]:
        count = bucket_requests.get(b, 0)
        lines.append(f"- {b}: {count} requests")

    lines.extend([
        "",
        "### Request Log Summary",
        f"- Total Requests in Log: {len(request_log)}",
    ])
    for entry in request_log:
        lines.append(f"- [{entry.get('bucket', 'all')}] status {entry.get('status_code')}, {entry.get('lines', 0)} lines, max={entry.get('max')}, cap_hit={entry.get('cap_hit', False)}")

    lines.extend([
        "",
        "### Ingestion Drops",
    ])
    if ingest_drop_stats:
        for reason, count in ingest_drop_stats.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (no drops recorded)")

    lines.extend([
        "",
        "## Extraction Drops"
    ])
    extraction_drops = dict(info.get("drop_stats", {}))
    for key in ["unparseable_pgn", "illegal_move", "no_target_moves"]:
        extraction_drops.setdefault(key, 0)
    for reason, count in sorted(extraction_drops.items()):
        lines.append(f"- {reason}: {count}")

    out_path = storage.player_dir(username) / "dataset_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
