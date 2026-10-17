"""Background worker stub (Phase I)."""

import time


def main() -> None:
    print("Starting Doppelgambit background worker...")
    while True:
        print("Worker heartbeat...")
        time.sleep(60)


if __name__ == "__main__":
    main()
