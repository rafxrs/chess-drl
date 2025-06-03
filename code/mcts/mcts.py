import chess
import numpy as np
import torch
import config
from utils import move_to_index
from mcts.node import Node
from mcts.edge import Edge

class MCTS:
    def __init__(self, agent, config_dict):
        self.agent = agent
        self.c_puct = config_dict.get('C_init', config.C_init)
        self.n_simulations = config_dict.get('SIMULATIONS_PER_MOVE', config.SIMULATIONS_PER_MOVE)
        self.dirichlet_alpha = config_dict.get('DIRICHLET_NOISE', config.DIRICHLET_NOISE)
        self.root = Node(None, 1.0)

    def run_simulation(self, model, n=None):
        if n is None:
            n = self.n_simulations
        for sim in range(n):
            try:
                state = chess.Board(self.agent.state)
                node = self.root
                search_path = [node]

                # Selection
                while not node.is_leaf():
                    action, edge = max(node.children.items(), key=lambda item: item[1].get_value(self.c_puct))
                    if action not in state.legal_moves:
                        # Skip this path if the move is no longer legal
                        break
                    node = edge.child
                    state.push(action)
                    search_path.append(node)

                # Only continue if we didn't break out of the selection loop
                if node == search_path[-1]:
                    # Expansion
                    if not state.is_game_over():
                        legal_moves = list(state.legal_moves)
                        state_tensor = self.agent.state_to_tensor(state)
                        with torch.no_grad():
                            policy_logits, _ = model(state_tensor)
                            policy = policy_logits.softmax(dim=1).cpu().numpy().flatten()
                        
                        # Filter for legal moves only
                        action_priors = [(move, policy[move_to_index(move)]) for move in legal_moves if move_to_index(move) < len(policy)]
                        
                        # Add Dirichlet noise at the root node only
                        if node is self.root and action_priors:
                            epsilon = 0.25  # AlphaZero default
                            alpha = self.dirichlet_alpha
                            noise = np.random.dirichlet([alpha] * len(action_priors))
                            action_priors = [
                                (move, (1 - epsilon) * p + epsilon * n)
                                for (move, p), n in zip(action_priors, noise)
                            ]

                        node.expand(action_priors, Node, Edge)

                    # Evaluation
                    value = self.evaluate_state(state, model)

                    # Backpropagation
                    for node in reversed(search_path):
                        node.update_recursive(value)
                        value = -value
            
            except Exception as e:
                # Log the error but continue with other simulations
                print(f"Error in simulation {sim}: {e}")
                continue

    def evaluate_state(self, state, model):
        if state.is_game_over():
            result = state.result()
            if result == '1-0':
                return 1
            elif result == '0-1':
                return -1
            else:
                return 0
        state_tensor = self.agent.state_to_tensor(state)
        with torch.no_grad():
            _, value = model(state_tensor)
            value = value.mean().item()
        return value

    def get_move_probs(self, temp=1.0):
        """
        Get the normalized visit counts for all possible moves.
        
        Args:
            temp: Temperature parameter controlling exploration
            
        Returns:
            actions: List of valid actions (chess.Move objects)
            probs: Corresponding probabilities for each action
        """
        actions = []
        visits = []
        
        # Collect visit counts
        for action, edge in self.root.children.items():
            actions.append(action)  # Use the action from the dictionary key
            if hasattr(edge, 'child') and hasattr(edge.child, 'visits'):
                visits.append(edge.child.visits)  # Use visits from the child Node
            elif hasattr(edge, 'N'):
                visits.append(edge.N) 
            else:
                # Fallback - if we can't find a visit count, use 1
                visits.append(1)
        
        visits = np.array(visits, dtype=np.float64)
        
        # Add a small epsilon to prevent zeros
        visits = visits + 1e-10
        
        # Normalize by temperature, safely handling large numbers
        if temp != 0:
            # Use log-sum-exp trick to prevent overflow
            log_visits = np.log(visits)
            log_visits = log_visits / temp
            log_visits = log_visits - np.max(log_visits)  # For numerical stability
            visits = np.exp(log_visits)
        
        # Normalize to get probabilities
        sum_visits = np.sum(visits)
        if sum_visits > 0:
            probs = visits / sum_visits
        else:
            # Fallback: uniform distribution
            probs = np.ones_like(visits) / len(visits)
        
        # Final safety check
        if np.isnan(probs).any() or not np.isclose(np.sum(probs), 1.0):
            # Fallback to uniform distribution
            probs = np.ones_like(visits) / len(visits)
        
        return actions, probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move].child
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)