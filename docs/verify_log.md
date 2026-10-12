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
