"""
Streaming reader for Lichess PGN archives (.pgn.zst) and files/URLs.
Never decompresses full archives to disk; streams compressed games on-the-fly.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, Union
import chess.pgn
import zstandard as zstd

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_USER_AGENT = "EchoChess-base-data/1.0 (+https://github.com/<owner>/dynamic_bots; contact: <owner-email>)"


def _normalize_date_str(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    val = str(val).strip().replace(".", "-")
    return val if val else None


def stream_games(
    path_or_url: Union[str, Path, io.IOBase],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Iterator[chess.pgn.Game]:
    """
    Stream chess.pgn.Game objects from a .pgn.zst file or URL without decompressing to disk.
    
    Parameters
    ----------
    path_or_url : str, Path, or IOBase
        Local file path, URL, or stream pointing to a .pgn.zst or .pgn source.
    start_date : str, optional
        Filter games on or after this date (inclusive, YYYY-MM-DD or YYYY.MM.DD).
    end_date : str, optional
        Filter games on or before this date (inclusive, YYYY-MM-DD or YYYY.MM.DD).

    Yields
    ------
    chess.pgn.Game
        Parsed chess game objects. Caller is responsible for filtering.
    """
    start_d = _normalize_date_str(start_date)
    end_d = _normalize_date_str(end_date)

    opened_resource = False
    raw_stream = None

    if isinstance(path_or_url, (io.RawIOBase, io.BufferedIOBase, io.TextIOBase)):
        raw_stream = path_or_url
    else:
        src = str(path_or_url)
        if src.startswith("http://") or src.startswith("https://"):
            req = urllib.request.Request(
                src,
                headers={"User-Agent": _USER_AGENT},
            )
            raw_stream = urllib.request.urlopen(req)
            opened_resource = True
        else:
            raw_stream = open(src, "rb")
            opened_resource = True

    try:
        # Determine whether input is compressed with zstandard
        if isinstance(raw_stream, io.TextIOBase):
            text_stream = raw_stream
        else:
            # Peek first bytes to detect zstd magic header or check filename
            is_zstd = False
            if hasattr(raw_stream, "peek"):
                try:
                    peek_bytes = raw_stream.peek(4)[:4]
                    is_zstd = peek_bytes.startswith(ZSTD_MAGIC)
                except Exception:
                    is_zstd = False
            elif hasattr(raw_stream, "seekable") and raw_stream.seekable():
                try:
                    pos = raw_stream.tell()
                    peek_bytes = raw_stream.read(4)
                    raw_stream.seek(pos)
                    is_zstd = peek_bytes.startswith(ZSTD_MAGIC)
                except Exception:
                    is_zstd = False

            if not is_zstd and not isinstance(path_or_url, io.IOBase) and str(path_or_url).endswith(".zst"):
                is_zstd = True

            if is_zstd:
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(raw_stream)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            else:
                text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8", errors="replace")

        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            if start_d is not None or end_d is not None:
                raw_date = game.headers.get("UTCDate") or game.headers.get("Date")
                g_d = _normalize_date_str(raw_date)
                if g_d:
                    if start_d is not None and g_d < start_d:
                        continue
                    if end_d is not None and g_d > end_d:
                        continue

            yield game

    finally:
        if opened_resource and raw_stream is not None:
            raw_stream.close()
