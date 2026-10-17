"""Run self-play match between bot and baseline opponent (Phase SP)."""

import argparse
import json
from pathlib import Path
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.analysis.self_play import load_pooled_think_time, play_match
from app.engine.model_cache import ModelCache
from app.engine.provider_v3 import V3Provider
from app.engine.think_time import ThinkTimeModel


def main():
    parser = argparse.ArgumentParser(description="Run self-play games between bot and opponent")
    parser.add_argument("--bot-username", type=str, default="the_king_crusher")
    parser.add_argument("--bot-version", type=int, default=1)
    parser.add_argument("--opponent-mode", type=str, default="base")
    parser.add_argument("--games", type=int, default=config.SP_DEFAULT_GAMES)
    parser.add_argument("--base-seconds", type=int, default=180)
    parser.add_argument("--increment-seconds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    data_root = Path(config.DATA_ROOT)
    player_dir = data_root / "players" / args.bot_username

    # 1. Determine player rating
    bot_rating = config.DEFAULT_RATING
    info_file = player_dir / "info.json"
    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
            bot_rating = int(info.get("median_rating", info.get("player_rating", config.DEFAULT_RATING)))

    # 2. Load bot think time model
    samples_path = player_dir / "samples_v3.parquet"
    if not samples_path.exists():
        samples_path = player_dir / "samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"No samples found for {args.bot_username} in {player_dir}")
    bot_tt = ThinkTimeModel.from_parquet(samples_path)

    # 3. Load opponent think time model from pooled base think time
    rating_band = int(round(bot_rating / 200.0) * 200)
    pooled_path = config.BASE_THINKTIME_DIR / "pooled_thinktime.json"
    if not pooled_path.exists():
        raise FileNotFoundError(f"Pooled thinktime file not found at {pooled_path}")
    opp_tt = load_pooled_think_time(pooled_path, rating_band=rating_band)

    # 4. Build shared model cache and providers
    cache = ModelCache(capacity=8, data_root=data_root)
    bot_provider = V3Provider(
        username=args.bot_username,
        model_version=args.bot_version,
        model_cache=cache,
        think_time_model=bot_tt,
        data_root=data_root,
    )
    opp_provider = V3Provider(
        username=args.opponent_mode,
        model_version=1,
        model_cache=cache,
        think_time_model=opp_tt,
        data_root=data_root,
    )

    # 5. Play match
    records = play_match(
        bot=bot_provider,
        opponent=opp_provider,
        n_games=args.games,
        base_s=args.base_seconds,
        inc_s=args.increment_seconds,
        seed=args.seed,
        bot_colors="alternate",
        bot_rating=bot_rating,
        opp_rating=bot_rating,
    )

    # 6. Analyze results
    bot_wins = 0
    opp_wins = 0
    draws = 0
    total_moves = 0
    bot_think_times = []
    opp_think_times = []

    for r in records:
        total_moves += len(r.moves)
        for m in r.moves:
            if m.by == "bot":
                bot_think_times.append(m.think_ms)
            else:
                opp_think_times.append(m.think_ms)

        if r.result == "1/2-1/2":
            draws += 1
        elif (r.result == "1-0" and r.bot_color == "white") or (r.result == "0-1" and r.bot_color == "black"):
            bot_wins += 1
        else:
            opp_wins += 1

    win_rate = (bot_wins / args.games) if args.games > 0 else 0.0
    avg_moves = (total_moves / args.games) if args.games > 0 else 0.0
    avg_bot_think = (sum(bot_think_times) / len(bot_think_times)) if bot_think_times else 0.0
    avg_opp_think = (sum(opp_think_times) / len(opp_think_times)) if opp_think_times else 0.0

    output_payload = {
        "metadata": {
            "bot": f"{args.bot_username}:v{args.bot_version}",
            "opponent": args.opponent_mode,
            "games": args.games,
            "base_s": args.base_seconds,
            "inc_s": args.increment_seconds,
            "seed": args.seed,
        },
        "results": {
            "bot_wins": bot_wins,
            "opp_wins": opp_wins,
            "draws": draws,
            "win_rate": round(win_rate, 4),
        },
        "games": [g.to_dict() for g in records],
    }

    # 7. Write to output file if specified
    out_path = args.out
    if out_path is None:
        out_path = (
            player_dir
            / "self_play"
            / f"v{args.bot_version}_vs_{args.opponent_mode}_{args.base_seconds}_{args.increment_seconds}.json"
        )
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # 8. Print summary table to stdout
    print(
        f"Result: {bot_wins} wins, {opp_wins} losses, {draws} draw"
        + ("s" if draws != 1 else "")
        + f" (win rate {win_rate * 100:.1f}%)"
    )
    print(f"Avg moves/game: {avg_moves:.1f}")
    print(f"Avg bot think (ms): {avg_bot_think:.0f}")
    print(f"Avg opp think (ms): {avg_opp_think:.0f}")


if __name__ == "__main__":
    main()
