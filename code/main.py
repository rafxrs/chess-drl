import argparse
import logging
import numpy as np
from game import Game
from agent import Agent
from env import Chess_Env
from gui.display import gui

logging.basicConfig(level=logging.INFO, format=" %(message)s")
logging.disable(logging.WARN)

class Main:
    def __init__(self, player_is_white: bool, model_path: str = None):
        self.player_is_white = player_is_white
        self.model_path = model_path
        self.env = Chess_Env()
        self.white_agent = Agent(model_path=model_path)
        self.black_agent = Agent(model_path=model_path)
        self.game = Game(self.env, self.white_agent, self.black_agent)
        self.GUI = gui(self.game)

    def loop(self):
        while not self.game.is_over():
            self.GUI.draw()
            if self.game.current_player_is_white() == self.player_is_white:
                move = self.get_player_move()
                self.game.push(move)
            else:
                move = self.opponent_move()
                self.game.push(move)
        self.GUI.draw()
        print("Game over:", self.game.result())

    def play_game(self):
        self.loop()

    def get_player_move(self):
        move = None
        while move not in self.game.legal_moves():
            move_str = input("Enter your move (in UCI format, e.g. e2e4): ")
            try:
                move = self.game.parse_move(move_str)
            except Exception:
                print("Invalid move format.")
        return move

    def opponent_move(self):
        # Use the agent to select a move
        agent = self.white_agent if not self.player_is_white else self.black_agent
        agent.run_simulations(n=100)
        actions, probs = agent.mcts.get_move_probs()
        move = np.random.choice(actions, p=probs)
        print(f"Agent plays: {move}")
        return move

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess DRL Main")
    parser.add_argument("--player", type=str, default=None, choices=('white', 'black'), help="Play as white or black. No argument means random.")
    parser.add_argument("--model", type=str, default=None, help="The path to the model to use.")
    args = parser.parse_args()
    
    model_path = args.model

    if args.player:
        player_is_white = args.player.lower().strip() == 'white'
    else:
        player_is_white = np.random.choice([True, False])

    m = Main(player_is_white, model_path)
    m.play_game()