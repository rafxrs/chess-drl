import chess
import numpy as np
import torch

import config
from utils import move_to_index


class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # move: Edge
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                child_node = Node(parent=self, prior_p=prob)
                self.children[action] = Edge(action, self, child_node, prob)

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class Edge:
    def __init__(self, move, parent, child, prior_p):
        self.move = move
        self.parent = parent
        self.child = child
        self.P = prior_p

    def get_value(self, c_puct):
        self.child.u = c_puct * self.P * (self.parent.n_visits ** 0.5) / (1 + self.child.n_visits)
        return self.child.Q + self.child.u


class MCTS:
    def __init__(self, agent, config_dict):
        self.agent = agent
        self.c_puct = config_dict.get("C_init", config.C_init)
        self.n_simulations = config_dict.get("SIMULATIONS_PER_MOVE", config.SIMULATIONS_PER_MOVE)
        self.dirichlet_alpha = config_dict.get("DIRICHLET_NOISE", config.DIRICHLET_NOISE)
        self.dirichlet_epsilon = config_dict.get("DIRICHLET_EPSILON", config.DIRICHLET_EPSILON)
        self.root = Node(None, 1.0)

    def run_simulation(self, model, n=None):
        n = n or self.n_simulations
        for _ in range(n):
            try:
                self._simulate_once(model)
            except Exception as e:
                print(f"Error in MCTS simulation: {e}")

    def _simulate_once(self, model):
        state = chess.Board(self.agent.state)
        node = self.root
        search_path = [node]

        # Selection
        while not node.is_leaf():
            action, edge = max(node.children.items(), key=lambda item: item[1].get_value(self.c_puct))
            if action not in state.legal_moves:
                return
            node = edge.child
            state.push(action)
            search_path.append(node)

        # Expansion
        if not state.is_game_over():
            legal_moves = list(state.legal_moves)
            state_tensor = self.agent.state_to_tensor(state).to(self.agent.device)
            with torch.no_grad():
                policy_logits, _ = model(state_tensor)
                policy = policy_logits.softmax(dim=1).cpu().numpy().flatten()

            action_priors = [(move, policy[move_to_index(move)]) for move in legal_moves if move_to_index(move) < len(policy)]

            if node is self.root and action_priors:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(action_priors))
                epsilon = self.dirichlet_epsilon
                action_priors = [(move, (1 - epsilon) * p + epsilon * eta) for (move, p), eta in zip(action_priors, noise)]

            node.expand(action_priors)

        # Evaluation
        value = self.evaluate_state(state, model)

        # Backpropagation
        for path_node in reversed(search_path):
            path_node.update_recursive(value)
            value = -value

    def evaluate_state(self, state, model):
        if state.is_game_over():
            result = state.result()
            if result == "1-0":
                return 1
            elif result == "0-1":
                return -1
            return 0

        state_tensor = self.agent.state_to_tensor(state).to(self.agent.device)
        with torch.no_grad():
            _, value = model(state_tensor)
        return value.mean().item()

    def get_move_probs(self, temp=1.0):
        """
        Get the normalized visit counts for all possible moves.

        Args:
            temp: Temperature parameter controlling exploration

        Returns:
            actions: List of valid actions (chess.Move objects)
            probs: Corresponding probabilities for each action
        """
        actions = list(self.root.children.keys())
        visits = np.array([edge.child.n_visits for edge in self.root.children.values()], dtype=np.float64)
        visits += 1e-10  # avoid zeros

        if temp != 0:
            # log-sum-exp trick to avoid overflow when raising visits to 1/temp
            log_visits = np.log(visits) / temp
            log_visits -= np.max(log_visits)
            visits = np.exp(log_visits)

        sum_visits = np.sum(visits)
        probs = visits / sum_visits if sum_visits > 0 else np.ones_like(visits) / len(visits)

        if np.isnan(probs).any() or not np.isclose(np.sum(probs), 1.0):
            probs = np.ones_like(visits) / len(visits)

        return actions, probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move].child
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
