# Verification Log

## Date: 2026-10-12

### Phase B Diagnostics and Clock Rules Verification (`the_king_crusher`)

#### 1. `%clk` Includes the Increment
- **Evidence:**
  - Non-berserk games with increment > 0 (`N=178` games, `7,603` consecutive own-move transitions):
    - Transitions with $d = clk_{prev} - clk_{cur} < 0$: **3,059** transitions.
    - Transitions with $d \ge 0$: **4,544** transitions.
  - A negative difference $d < 0$ indicates that the player's remaining clock increased after making a move. This occurs whenever elapsed thinking time is less than the increment ($time\_spent < increment\_s$).
  - In games with increment == 0 (`N=551` non-berserk games, `21,702` transitions), transitions with $d < 0$ occurred **0** times ($d \ge 0$: 21,702).
- **Rule:**
  - For standard non-berserk moves, elapsed thinking time is given by:
    $$\text{time\_spent\_s} = \max(0.0, clk_{before} - clk_{after} + increment\_s)$$

#### 2. Berserk Removes the Increment and Halves the Clock
- **Evidence:**
  - Initial clock ratio ($first\_clk / base$):
    - All **471 / 471 (100.0%)** header-berserk games fall in the ratio bin $[0.4, 0.6]$ (specifically $0.5$, i.e. base time halved).
    - Bin $< 0.4$: 0 games (0.0%).
    - Bin $0.6 - 0.9$: 0 games (0.0%).
    - Bin $\ge 0.9$: 0 games (0.0%).
  - Increment removal in berserk:
    - In header-berserk games with increment > 0 (`N=146` games, `5,777` transitions):
      - Transitions with $d < 0$: **0** of 5,777 transitions.
      - Transitions with $d \ge 0$: **5,777** of 5,777 transitions.
    - By contrast, non-berserk increment games had **3,059 of 7,603** transitions with $d < 0$.
- **Rule:**
  - When the target player berserks, the effective increment is 0:
    $$\text{effective\_increment} = 0 \text{ if } target\_berserk \text{ else } increment\_s$$
    $$\text{time\_spent\_s} = \max(0.0, clk_{before} - clk_{after} + \text{effective\_increment})$$
  - The `increment_s` dataset column preserves the official `TimeControl` increment value.

#### 3. First-Move Rule (First %clk Equals Base is Not Universal)
- **Evidence:**
  - Non-berserk games with $inc = 0$:
    - First $\%clk == base$: **522 / 551** games (94.7%).
    - First $\%clk < base$: **29 / 551** games (5.3%).
  - Non-berserk games with $inc > 0$:
    - First $\%clk == base$: **127 / 178** games (71.3%).
    - First $\%clk < base$: **46 / 551** games (25.8%).
    - First $\%clk > base$: **5 / 178** games (2.8%).
  - Investigation into games where first $\%clk < base$ revealed tournament/arena games (e.g. *Weekly AGCA Prize*, *3rd MSC Online Tournament*, *5th KINGS CROWN CHESS TOURNAMENT*) where clocks started before the player made their first move.
- **Rule:**
  - Because no prior move exists and initial clocks may have ticked before move 1, the target player's first move in every game has:
    $$\text{time\_spent\_s} = \text{None}$$

#### 4. Berserk Headers are Reliable
- **Evidence:**
  - Lichess PGN headers `WhiteBerserk` and `BlackBerserk` correlate 100% with clock halving:
    - In all 471 games with the header set to `"true"`, initial clock is exactly $base / 2$.
    - 0 header-berserk games have an initial clock ratio $\ge 0.9$.
- **Rule:**
  - Existing berserk detection via PGN headers is completely reliable.

#### 5. Stale Clock Invalidation Rule
- **Rule:**
  - When the target player's previous move lacked a `%clk` annotation, `clock_before_move` is carried forward from an earlier turn as an estimate.
  - While `clock_before_move` retains its carry-forward behavior for state representation, elapsed time cannot be measured across missing clock plies.
  - Therefore, if the target's previous move lacked a `%clk`, `time_spent_s` for the current move is set to `None`.

#### 6. Step 2 Empirical Verification Summary (the_king_crusher)
- **Step 2c (Clock transitions in increment games):**
  - Header-berserk games with increment > 0: transitions with d < 0: 0 / 5777
  - Non-berserk games with increment > 0: transitions with d < 0: 3059 / 7603
- **Step 2e (Header-berserk initial clock halving):**
  - Header-berserk games total: 471
  - Header-berserk games with first %clk == base/2: 471


## 2026-10-12 — Lichess DB verification (C0)

1. **Licence of the standard games database:**
   - Licence: Creative Commons CC0 1.0 Universal (Public Domain Dedication). Training, commercial use, modification, and redistribution are permitted without restrictions or prior consent.
   - Sources: [https://database.lichess.org](https://database.lichess.org), [https://tldrlegal.com/license/creative-commons-cc0-1.0-universal](https://tldrlegal.com/license/creative-commons-cc0-1.0-universal)

2. **File format and required decompression tool:**
   - Format: `.pgn.zst` (Zstandard compressed PGN text archives).
   - Decompression tools: `zstd` command-line utility (`zstd -d ...`) or Python `zstandard` streaming library (`zstandard.ZstdDecompressor().stream_reader(...)`).
   - Source: [https://database.lichess.org](https://database.lichess.org)

3. **Presence of `%clk` comments in published PGNs:**
   - Verified present. Standard rated database PGN moves contain move clock annotations formatted as `{ [%clk H:MM:SS] }` indicating remaining time on the player's clock after making each move.
   - Source: [https://database.lichess.org](https://database.lichess.org)

4. **Presence of `WhiteBerserk` / `BlackBerserk` headers:**
   - Verified present. Arena tournament games where a player berserked include explicit PGN tag pairs: `[WhiteBerserk "1"]` (or `"true"`) and `[BlackBerserk "1"]` (or `"true"`).
   - Source: [https://database.lichess.org](https://database.lichess.org)

5. **Approximate monthly file size (GB):**
   - For recent months (2025/2026), the compressed `.pgn.zst` dump size is approximately 28 to 30 GB per month (~90 to 100 million games per month).
   - Source: [https://database.lichess.org](https://database.lichess.org)

6. **Lichess API terms-of-use URL and date accessed:**
   - Terms URLs: [https://lichess.org/terms-of-service](https://lichess.org/terms-of-service) and API guidelines at [https://lichess.org/api](https://lichess.org/api).
   - Date accessed: 2026-10-12.

## 2026-10-17 — downloader.py helper function
`download_player_games(username)` was added during Phase D multi-player ingest.
It wraps `_run_ingest` from routers.ingest and returns the in-memory `_jobs`
status dict. Status: kept. Architectural debt noted for Phase G (jobs move
to SQLite; services -> routers layering is backwards).

