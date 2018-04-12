import collections


class RootNode(object):
    def __init__(self):
        self.parent = None
        self.child_visit_count = collections.default_dict(float)
        self.child_total_reward = collections.default_dict(float)


class MCTSTreeNode(object):
    def __init__(self, game, state, player, parent=None, action=None):
        self.parent = parent if parent else RootNode()
        self.game = game
        self.state = state
        self.player = player
        self.action = action
