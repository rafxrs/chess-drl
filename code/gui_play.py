# Play against a specific bot checkpoint with a graphical chess board.
# Pick any saved version, e.g. --model models/model_iter_5.pt, to see how
# strong the bot was at that point in training.
import argparse
import logging
import time

import numpy as np

import config
from agent import Agent
from env import Chess_Env
from gui.display import GUI

logging.basicConfig(level=logging.WARNING, format=" %(message)s")

WINDOW_SIZE = 800


def main():
    parser = argparse.ArgumentParser(description="Play chess against a specific bot checkpoint, with a graphical board")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH, help="Path to the model checkpoint to play against")
    parser.add_argument("--color", choices=["white", "black", "random"], default="random")
    parser.add_argument("--simulations", type=int, default=config.SIMULATIONS_PER_MOVE)
    args = parser.parse_args()

    human_is_white = np.random.choice([True, False]) if args.color == "random" else args.color == "white"

    print(f"Loading bot from {args.model} ...")
    agent = Agent(model_path=args.model)
    agent.mcts.n_simulations = args.simulations

    env = Chess_Env()
    gui = GUI(width=WINDOW_SIZE, height=WINDOW_SIZE, player=True, fen=env.board.fen())
    gui.gameboard.board = env.board  # share the same board so human clicks and bot moves both land on env.board

    print(f"You are playing {'White' if human_is_white else 'Black'}. Close the window or press Esc to quit.")

    while not env.is_game_over():
        gui.draw()

        if env.board.turn != human_is_white:
            move = agent.get_move(env)
            env.push(move)
            gui.draw()
            time.sleep(0.2)

    gui.draw()
    print(f"Game over: {env.get_result()}")
    time.sleep(5)


if __name__ == "__main__":
    main()
