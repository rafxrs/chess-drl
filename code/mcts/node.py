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

    def expand(self, action_priors, node_class, edge_class):
        for action, prob in action_priors:
            if action not in self.children:
                child_node = node_class(parent=self, prior_p=prob)
                self.children[action] = edge_class(action, self, child_node, prob)

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)