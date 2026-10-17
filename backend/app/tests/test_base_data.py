"""
Tests for Phase C.1 base data pipeline:
streaming, filtering, cohort decontamination, band balancing, and berserk heuristic.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
import chess.pgn
import pytest

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.base_data.filter import passes_filters, is_decontaminated, parse_tc, classify_tc
from tools.base_data.stream_pgn import stream_games
from tools.base_data.build_shards import detect_berserk, build_shards, get_rating_band


def make_pgn_game_str(
    site_id: str = "game1",
    event: str = "Rated Blitz game",
    variant: str = "Standard",
    tc: str = "180+0",
    termination: str = "Normal",
    white: str = "PlayerA",
    black: str = "PlayerB",
    white_elo: str = "1500",
    black_elo: str = "1500",
    date: str = "2026.01.15",
    plies: int = 10,
    with_clks: bool = True,
    setup: str = "0",
    fen: str | None = None,
    white_berserk: str | None = None,
    black_berserk: str | None = None,
    white_first_clk: float = 180.0,
    black_first_clk: float = 180.0,
) -> str:
    """Helper to construct small synthetic PGN games."""
    headers = [
        f'[Event "{event}"]',
        f'[Site "https://lichess.org/{site_id}"]',
        f'[Date "{date}"]',
        f'[UTCDate "{date}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        f'[Result "1-0"]',
        f'[WhiteElo "{white_elo}"]',
        f'[BlackElo "{black_elo}"]',
        f'[Variant "{variant}"]',
        f'[TimeControl "{tc}"]',
        f'[Termination "{termination}"]',
    ]
    if setup == "1":
        headers.append('[SetUp "1"]')
    if fen:
        headers.append(f'[FEN "{fen}"]')
    if white_berserk is not None:
        headers.append(f'[WhiteBerserk "{white_berserk}"]')
    if black_berserk is not None:
        headers.append(f'[BlackBerserk "{black_berserk}"]')

    # Moves alternating e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7
    move_pairs = [
        ("e4", "e5"),
        ("Nf3", "Nc6"),
        ("Bb5", "a6"),
        ("Ba4", "Nf6"),
        ("O-O", "Be7"),
        ("d3", "d6"),
    ]

    body_moves = []
    current_ply = 0
    w_clock = white_first_clk
    b_clock = black_first_clk

    for w_move, b_move in move_pairs:
        if current_ply < plies:
            clk_str = f" {{ [%clk 0:{int(w_clock//60):02d}:{int(w_clock%60):02d}] }}" if with_clks else ""
            body_moves.append(f"{(current_ply // 2) + 1}. {w_move}{clk_str}")
            w_clock = max(1.0, w_clock - 2.0)
            current_ply += 1
        if current_ply < plies:
            clk_str = f" {{ [%clk 0:{int(b_clock//60):02d}:{int(b_clock%60):02d}] }}" if with_clks else ""
            body_moves.append(f"{b_move}{clk_str}")
            b_clock = max(1.0, b_clock - 2.0)
            current_ply += 1

    body = " ".join(body_moves) + " 1-0\n"
    return "\n".join(headers) + "\n\n" + body + "\n"


@pytest.fixture
def pgn_200_games_file(tmp_path: Path) -> Path:
    """
    Generate a 200-game PGN fixture file with:
      - Exactly 100 valid games (pass all filters).
      - Exactly 100 invalid games failing specific filters.
    """
    lines: list[str] = []

    # 1. 100 valid games
    for i in range(100):
        lines.append(make_pgn_game_str(
            site_id=f"valid_{i:03d}",
            event="Rated Blitz game",
            variant="Standard",
            tc="180+0",
            termination="Normal",
            white=f"ValidWhite_{i}",
            black=f"ValidBlack_{i}",
            white_elo="1500",
            black_elo="1600",
            plies=10,
            with_clks=True,
        ))

    # 2. 100 invalid games:
    # 15 non-standard variant
    for i in range(15):
        lines.append(make_pgn_game_str(site_id=f"inv_var_{i}", variant="Chess960"))
    # 15 non-rated event
    for i in range(15):
        lines.append(make_pgn_game_str(site_id=f"inv_evt_{i}", event="Casual Blitz game"))
    # 10 custom FEN
    for i in range(10):
        lines.append(make_pgn_game_str(
            site_id=f"inv_fen_{i}",
            setup="1",
            fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        ))
    # 15 invalid time control (< 60s base or non-included category like classical)
    for i in range(15):
        tc_val = "30+0" if i % 2 == 0 else "1500+0"
        lines.append(make_pgn_game_str(site_id=f"inv_tc_{i}", tc=tc_val))
    # 15 bad termination
    for i in range(15):
        lines.append(make_pgn_game_str(site_id=f"inv_term_{i}", termination="Abandoned"))
    # 15 short games (< 10 plies)
    for i in range(15):
        lines.append(make_pgn_game_str(site_id=f"inv_ply_{i}", plies=6))
    # 10 missing/unparseable rating
    for i in range(10):
        lines.append(make_pgn_game_str(site_id=f"inv_elo_{i}", white_elo="?"))
    # 5 missing clocks
    for i in range(5):
        lines.append(make_pgn_game_str(site_id=f"inv_clk_{i}", with_clks=False))

    content = "".join(lines)
    pgn_path = tmp_path / "fixture_200_games.pgn"
    pgn_path.write_text(content, encoding="utf-8")
    return pgn_path


def test_1_passes_filters(pgn_200_games_file: Path):
    """
    Test 1: passes_filters accepts exactly the hand-counted set of games.
    Out of 200 fixture games, exactly 100 are valid and 100 are invalid.
    """
    games = list(stream_games(pgn_200_games_file))
    assert len(games) == 200, f"Expected 200 games streamed, got {len(games)}"

    passed = [g for g in games if passes_filters(g)]
    rejected = [g for g in games if not passes_filters(g)]

    assert len(passed) == 100, f"Expected exactly 100 passing games, got {len(passed)}"
    assert len(rejected) == 100, f"Expected exactly 100 rejected games, got {len(rejected)}"

    # Check that all accepted games have valid_ site prefix
    for g in passed:
        assert "valid_" in g.headers["Site"]


def test_2_band_balancing(tmp_path: Path):
    """
    Test 2: band × speed stratification balancing.
    Synthetic games split a single rating band into TWO speed categories
    (blitz TC=180+0 and bullet TC=60+0 at the same rating 1500).

    Hand-computed expectation:
      - Rating 1500 belongs to band "1400-1600".
      - 2 games with TC="180+0" (blitz): each 10-ply game generates 4 candidate
        positions (plies 4, 5, 8, 9), yielding 8 candidates for "1400-1600|blitz".
      - 2 games with TC="60+0" (bullet): each 10-ply game generates 4 candidate
        positions, yielding 8 candidates for "1400-1600|bullet".
      - With band_cap = 5, each (band, speed) cell is capped at exactly 5 positions:
          cell_counts["1400-1600|blitz"] == 5
          cell_counts["1400-1600|bullet"] == 5
      - Total balanced positions across both cells = 5 + 5 = 10 positions.
    """
    lines: list[str] = []
    # 2 blitz games (TC 180+0) at rating 1500
    for i in range(2):
        lines.append(make_pgn_game_str(
            site_id=f"strat_blitz_{i}",
            white_elo="1500",
            black_elo="1500",
            tc="180+0",
            plies=10,
            date="2026.01.10",
        ))
    # 2 bullet games (TC 60+0) at rating 1500
    for i in range(2):
        lines.append(make_pgn_game_str(
            site_id=f"strat_bullet_{i}",
            white_elo="1500",
            black_elo="1500",
            tc="60+0",
            plies=10,
            date="2026.01.10",
        ))

    pgn_path = tmp_path / "strat_games.pgn"
    pgn_path.write_text("".join(lines), encoding="utf-8")

    out_dir = tmp_path / "shards_out"
    config = {"shard_size": 1000, "band_cap": 5}
    report = build_shards(months=[pgn_path], out_dir=out_dir, config=config, cohort=[])

    assert report["total_games_streamed"] == 4
    assert report["games_passed_filters"] == 4

    expected_blitz_cell = "1400-1600|blitz"
    expected_bullet_cell = "1400-1600|bullet"

    assert expected_blitz_cell in report["cell_counts"], f"Missing {expected_blitz_cell}"
    assert expected_bullet_cell in report["cell_counts"], f"Missing {expected_bullet_cell}"
    assert report["cell_counts"][expected_blitz_cell] == 5, (
        f"Expected 5 positions for {expected_blitz_cell}, got {report['cell_counts'][expected_blitz_cell]}"
    )
    assert report["cell_counts"][expected_bullet_cell] == 5, (
        f"Expected 5 positions for {expected_bullet_cell}, got {report['cell_counts'][expected_bullet_cell]}"
    )
    assert sum(report["cell_counts"].values()) == 10


def test_3_berserk_heuristic():
    """
    Test 3: berserk heuristic on 4 synthetic games (headers vs. clock-based).
      Game 1: Header WhiteBerserk = "1" -> True (header override)
      Game 2: Header WhiteBerserk = "0" -> False (header override)
      Game 3: No header, base=180, inc=0; White first clk = 90.0s (<= 91.0) -> True
      Game 4: No header, base=180, inc=0; White first clk = 175.0s (> 91.0) -> False
    """
    # Game 1: Header WhiteBerserk = "1"
    g1_str = make_pgn_game_str(white_berserk="1", white_first_clk=180.0)
    g1 = chess.pgn.read_game(io.StringIO(g1_str))
    assert detect_berserk(g1, chess.WHITE, base_s=180, inc_s=0) is True

    # Game 2: Header WhiteBerserk = "0"
    g2_str = make_pgn_game_str(white_berserk="0", white_first_clk=80.0)
    g2 = chess.pgn.read_game(io.StringIO(g2_str))
    assert detect_berserk(g2, chess.WHITE, base_s=180, inc_s=0) is False

    # Game 3: Clock-based berserk (first %clk <= 180/2 + 0 + 1 = 91.0s)
    g3_str = make_pgn_game_str(white_first_clk=90.0)
    g3 = chess.pgn.read_game(io.StringIO(g3_str))
    assert detect_berserk(g3, chess.WHITE, base_s=180, inc_s=0) is True

    # Game 4: Clock-based non-berserk (first %clk = 175.0s > 91.0s)
    g4_str = make_pgn_game_str(white_first_clk=175.0)
    g4 = chess.pgn.read_game(io.StringIO(g4_str))
    assert detect_berserk(g4, chess.WHITE, base_s=180, inc_s=0) is False


def test_4_decontamination(tmp_path: Path):
    """
    Test 4: decontamination removes all fixture games whose White/Black
    header matches a cohort name.
    """
    cohort = ["the_king_crusher", "Carlsen"]

    # 4 synthetic games:
    # Game A: White is 'the_king_crusher' (contaminated)
    # Game B: Black is 'THE_KING_CRUSHER' (case-insensitive, contaminated)
    # Game C: White is 'Carlsen' (contaminated)
    # Game D: Clean players (decontaminated)
    games = [
        make_pgn_game_str(site_id="game_a", white="the_king_crusher", black="Opponent1"),
        make_pgn_game_str(site_id="game_b", white="Opponent2", black="THE_KING_CRUSHER"),
        make_pgn_game_str(site_id="game_c", white="Carlsen", black="Opponent3"),
        make_pgn_game_str(site_id="game_d", white="CleanWhite", black="CleanBlack"),
    ]

    pgn_path = tmp_path / "decontam_games.pgn"
    pgn_path.write_text("".join(games), encoding="utf-8")

    out_dir = tmp_path / "shards_decontam"
    report = build_shards(months=[pgn_path], out_dir=out_dir, cohort=cohort)

    assert report["total_games_streamed"] == 4
    assert report["games_passed_filters"] == 4
    assert report["games_decontaminated"] == 1, (
        f"Expected exactly 1 decontaminated game, got {report['games_decontaminated']}"
    )

    # Directly check is_decontaminated function
    g_objs = list(stream_games(pgn_path))
    assert not is_decontaminated(g_objs[0], cohort)
    assert not is_decontaminated(g_objs[1], cohort)
    assert not is_decontaminated(g_objs[2], cohort)
    assert is_decontaminated(g_objs[3], cohort)


import zstandard as zstd


def test_5_stream_games_from_zstd_file(tmp_path: Path):
    """
    Test 5: stream_games decodes a real .pgn.zst file and yields exactly
    the games that were compressed.
    Build 3 games of 10 plies each (as produced by make_pgn_game_str),
    compress with zstandard.ZstdCompressor().compress(), write to
    tmp_path/archive.pgn.zst, and assert:
      - len(list(stream_games(path))) == 3
      - headers of game 0 match the original
      - pass an io.BytesIO stream of the compressed bytes as well and
        assert the same result
    """
    games_str = "".join(
        make_pgn_game_str(site_id=f"zstd_{i:02d}") for i in range(3)
    )
    compressed = zstd.ZstdCompressor().compress(games_str.encode("utf-8"))

    pgn_zst = tmp_path / "archive.pgn.zst"
    pgn_zst.write_bytes(compressed)

    # Case A: file path
    g_list = list(stream_games(pgn_zst))
    assert len(g_list) == 3
    assert g_list[0].headers["Site"].endswith("zstd_00")

    # Case B: BytesIO stream
    buf = io.BytesIO(compressed)
    g_list2 = list(stream_games(buf))
    assert len(g_list2) == 3
    assert g_list2[2].headers["Site"].endswith("zstd_02")


def test_6_stream_games_url_zstd_magic_check(tmp_path: Path):
    """
    Test 6: verify zstd detection when path does NOT end in .zst but
    the content magic matches. This guards the peek-based detection.
    Compress 2 games, write to tmp_path/archive_bytes.txt (no .zst ext),
    assert len(list(stream_games(path))) == 2.
    """
    games_str = "".join(
        make_pgn_game_str(site_id=f"magic_{i}") for i in range(2)
    )
    compressed = zstd.ZstdCompressor().compress(games_str.encode("utf-8"))
    p = tmp_path / "archive_bytes.txt"
    p.write_bytes(compressed)
    g_list = list(stream_games(p))
    assert len(g_list) == 2
    assert g_list[0].headers["Site"].endswith("magic_0")

