class Game(object):
    """
    Base Game class for two-player, zero-sum,
    turn-based games.
    """

    @staticmethod
    def initial_state():
        """
        The initial state of the game when no
        moves have been played.

        Returns:
            state: the initial state of the game
        """
        raise NotImplementedError

    @staticmethod
    def actions(state):
        """
        A list of all actions that are available to
        the player at the current game state.

        Args:
            state: current state of the game
            player: current player playing (1 or -1)
        Returns:
            All list available actions
        """
        raise NotImplementedError

    @staticmethod
    def result(state, action):
        """
        The resulting state that is produced when the
        player applies the action to the game state.
        In turn-based games the resulting player
        is the opponent (-player)

        Args:
            state: current state of the game
            action: action taken by current player
        Returns:
            result_state: resulting state of the game
        """
        raise NotImplementedError

    @staticmethod
    def terminal_test(state):
        """
        Check if the game has ended and if yes determine
        the outcome of game.

        Args:
            state: current state of the game
        Returns:
            0 if game has not ended
            1 if player won, -1 if player lost
        """
        raise NotImplementedError
