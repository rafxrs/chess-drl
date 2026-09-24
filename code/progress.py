# Visualize how the bot is learning: training loss and the candidate's win
# rate against the previous best model, iteration by iteration. Can be run
# while train.py is still going, with --watch to keep refreshing.
import argparse
import csv
import os
import time

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import config


def load_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot(rows, save_path=None):
    if not rows:
        print("No training data logged yet. Start training with train.py first.")
        return

    iterations = [int(r["iteration"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    policy_loss = [float(r["policy_loss"]) for r in rows]
    value_loss = [float(r["value_loss"]) for r in rows]
    win_rate = [float(r["win_rate"]) * 100 for r in rows]
    promoted = [r["promoted"] == "True" for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(iterations, loss, label="Total loss")
    ax1.plot(iterations, policy_loss, label="Policy loss")
    ax1.plot(iterations, value_loss, label="Value loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss")
    ax1.legend()
    ax1.grid(True)

    colors = ["tab:green" if p else "tab:gray" for p in promoted]
    ax2.bar(iterations, win_rate, color=colors)
    ax2.axhline(y=config.WIN_RATE_THRESHOLD * 100, color="red", linestyle="--",
                label=f"Promotion threshold ({config.WIN_RATE_THRESHOLD:.0%})")
    ax2.set_ylabel("Win rate vs previous best (%)")
    ax2.set_xlabel("Iteration")
    ax2.set_title("Self-play learning progress (green = new best model promoted)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved progress plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss and win-rate progress over time")
    parser.add_argument("--log", type=str, default=config.TRAINING_LOG_PATH)
    parser.add_argument("--watch", action="store_true", help="Keep refreshing the plot as training progresses")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between refreshes in --watch mode")
    parser.add_argument("--save", type=str, default=None, help="Save the plot to this file instead of showing it interactively")
    args = parser.parse_args()

    headless = matplotlib.get_backend().lower() == "agg"
    save_path = args.save or (os.path.splitext(config.TRAINING_LOG_PATH)[0] + ".png" if headless else None)

    if not args.watch:
        plot(load_log(args.log), save_path=save_path)
        return

    if headless:
        print(f"No display detected: refreshing {save_path} every {args.interval:.0f}s. Open it in an image viewer to watch it update.")
    print("Watching for training progress... press Ctrl+C to stop.")
    try:
        while True:
            plt.close("all")
            plot(load_log(args.log), save_path=save_path)
            if headless:
                time.sleep(args.interval)
            else:
                plt.pause(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
