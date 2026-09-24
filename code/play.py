# Play a game against the bot in the terminal.
import argparse
import logging

import chess
import numpy as np

import config
from agent import Agent
from env import Chess_Env

logging.basicConfig(level=logging.WARNING, format=" %(message)s")


def prompt_move(env):
    while True:
        move_str = input("Your move (UCI, e.g. e2e4; or 'moves' to list legal moves): ").strip()
        if move_str.lower() in ("moves", "legal"):
            print(", ".join(m.uci() for m in env.legal_moves()))
            continue

        move = None
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            try:
                move = env.board.parse_san(move_str)
            except ValueError:
                pass

        if move is None:
            print("Could not parse that move. Try UCI format like e2e4.")
        elif move not in env.board.legal_moves:
            print("That move is not legal in this position.")
        else:
            return move


def main():
    parser = argparse.ArgumentParser(description="Play chess against the trained bot in the terminal")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH, help="Path to the model checkpoint to play against")
    parser.add_argument("--color", choices=["white", "black", "random"], default="random")
    parser.add_argument("--simulations", type=int, default=config.SIMULATIONS_PER_MOVE)
    args = parser.parse_args()

    human_is_white = np.random.choice([True, False]) if args.color == "random" else args.color == "white"

    print(f"Loading bot from {args.model} ...")
    agent = Agent(model_path=args.model)
    agent.mcts.n_simulations = args.simulations

    env = Chess_Env()
    print(f"You are playing {'White' if human_is_white else 'Black'}.\n")

    try:
        while not env.is_game_over():
            print(env.board)
            print()
            if env.board.turn == human_is_white:
                move = prompt_move(env)
            else:
                print("Bot is thinking...")
                move = agent.get_move(env)
                print(f"Bot plays: {move.uci()}")
            env.push(move)
            print()
    except (EOFError, KeyboardInterrupt):
        print("\nGame abandoned.")
        return

    print(env.board)
    print(f"\nGame over: {env.get_result()}")


if __name__ == "__main__":
    main()
