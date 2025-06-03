class Edge:
    def __init__(self, move, parent, child, prior_p):
        self.move = move
        self.parent = parent
        self.child = child
        self.P = prior_p

    def get_value(self, c_puct):
        # UCT value for edge selection
        self.child.u = (c_puct * self.P * (self.parent.n_visits ** 0.5) / (1 + self.child.n_visits))
        return self.child.Q + self.child.u