"""
Graph how training is going: loss, how self-play games end, and how long
they last. Run it alongside train.py; --watch keeps it refreshing.
"""
import argparse
import csv
import os
import sys
import time

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")  # no screen to draw on: save a PNG instead

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config


def load_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def column(rows, name):
    """Float values for a column; blank (rows from older versions) become NaN."""
    return [float(r[name]) if r.get(name) not in (None, "") else float("nan") for r in rows]


def plot(rows):
    """Draw the figure; returns False if there's nothing to plot yet."""
    if not rows:
        print("No training data logged yet. Start training with train.py first.")
        return False

    iterations = [int(r["iteration"]) for r in rows]
    games = column(rows, "games")
    share = lambda name: [100 * x / g if g else float("nan") for x, g in zip(column(rows, name), games)]

    fig, (ax_loss, ax_results, ax_length) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax_loss.plot(iterations, column(rows, "loss"), label="Total")
    ax_loss.plot(iterations, column(rows, "policy_loss"), label="Policy")
    ax_loss.plot(iterations, column(rows, "value_loss"), label="Value")
    ax_loss.set_title("Training loss (lower is better)")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_results.stackplot(
        iterations, share("white_wins"), share("black_wins"), share("draws"),
        labels=["White wins", "Black wins", "Draws"],
        colors=["#d9d9d9", "#404040", "#8fb3d9"],
    )
    ax_results.plot(iterations, share("adjudicated"), color="tab:red", linestyle="--", label="Decided by move limit")
    ax_results.set_title("How self-play games end")
    ax_results.set_ylabel("% of games")
    ax_results.set_ylim(0, 100)
    ax_results.legend(loc="upper left", fontsize="small")
    ax_results.grid(True)

    ax_length.plot(iterations, column(rows, "avg_plies"), color="tab:purple")
    ax_length.axhline(config.MAX_GAME_MOVES, color="tab:red", linestyle="--", label="Move limit")
    ax_length.set_title("Average game length")
    ax_length.set_ylabel("Plies")
    ax_length.set_xlabel("Iteration")
    ax_length.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_length.legend()
    ax_length.grid(True)

    plt.tight_layout()
    return True


def main():
    parser = argparse.ArgumentParser(description="Graph training loss and self-play results over time")
    parser.add_argument("--log", type=str, default=config.TRAINING_LOG_PATH)
    parser.add_argument("--watch", action="store_true", help="Keep refreshing the plot as training progresses")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between refreshes in --watch mode")
    parser.add_argument("--save", type=str, default=None, help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    headless = matplotlib.get_backend().lower() == "agg"
    save_path = args.save or (os.path.splitext(args.log)[0] + ".png" if headless else None)

    def render():
        plt.close("all")
        if plot(load_log(args.log)) and save_path:
            plt.savefig(save_path)
            print(f"Saved progress plot to {save_path}")

    if not args.watch:
        render()
        if not save_path:
            plt.show()
        return

    if headless:
        print(f"No display detected: refreshing {save_path} every {args.interval:.0f}s.")
    else:
        plt.ion()  # non-blocking window so the loop can redraw it
    print("Watching training progress... press Ctrl+C to stop.")
    try:
        while True:
            render()
            if headless:
                time.sleep(args.interval)
            else:
                plt.pause(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
