import datetime
from enum import Enum
import pathlib
import string
from typing import List, Optional, Tuple

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 7, 7)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)

        # Available actions of the game
        # Dimenions^4 because we need to specify the start and end position of the piece
        self.action_space = list(range(Hnefatafl.DIMENSION ** 4))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class


        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 300  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network


        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000


        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False


        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class PieceType(Enum):
    DEFENDER = 0
    ATTACKER = 1
    KING = 2

    def to_string(self) -> str:
        if self == PieceType.DEFENDER:
            return "🛡️"
        if self == PieceType.ATTACKER:
            return "🗡️"
        if self == PieceType.KING:
            return "K"


class Position:
    """
    Represents a zero-based position on the board.
    """

    MIDDLE = (3, 3)
    UPPER_LEFT = (0, 0)
    UPPER_RIGHT = (6, 0)
    LOWER_LEFT = (0, 6)
    LOWER_RIGHT = (6, 6)
    CORNERS = [UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT]
    RESTRICTED_POSITIONS = [UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT, MIDDLE]

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def up(self, steps: int = 1) -> "Position":
        return Position(self.x, self.y + steps)

    def left(self, steps: int = 1) -> "Position":
        return Position(self.x - steps, self.y)

    def right(self, steps: int = 1) -> "Position":
        return Position(self.x + steps, self.y)

    def down(self, steps: int = 1) -> "Position":
        return Position(self.x, self.y - steps)

    def get_square(self, board: List[List[Optional[PieceType]]]) -> Optional[PieceType]:
        if self.x < 0 or self.x >= len(board) or self.y < 0 or self.y >= len(board):
            return None
        return board[self.y][self.x]

    def set_square(
        self, board: List[List[Optional[PieceType]]], piece: Optional[PieceType]
    ):
        board[self.y][self.x] = piece

    def is_open_to_piece(self, piece: PieceType) -> bool:
        """
        Checks if a position is
            - on the board
            - not restricted for piece (only king can enter restricted positions)
        Returns true if position is open to given piece and false otherwise.
        """

        if piece is None:
            print("Piece is none")
            return False

        is_within_board = (
            0 <= self.x < Hnefatafl.DIMENSION and 0 <= self.y < Hnefatafl.DIMENSION
        )
        if not is_within_board:
            return False
        if self in Position.RESTRICTED_POSITIONS:
            return piece == PieceType.KING  # only king can enter restricted squares
        return True

    def to_string(self) -> str:
        return f"({chr(ord('A') + self.x)}{abs(self.y - Hnefatafl.DIMENSION)})"


class Piece:
    def __init__(self, piece_type: PieceType, position: Position):
        self.piece_type = piece_type
        self.position = position
        self.captured = False


Move = Tuple[Position, Position]


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Hnefatafl()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        action = 0
        while True:
            try:
                print(
                    f"Enter the start position of the *{self.env.current_player.to_string()}* piece you wish to move:"
                )
                start_col = input(
                    f"Enter the column (A - G) to play for the player {self.to_play()}: "
                )
                if "A" <= start_col <= "G":
                    start_col = ord(start_col) - ord(
                        "A"
                    )  # Convert letter to 0-based index
                else:
                    print("Invalid input! Please enter a letter between A and G.")
                    iter
                start_row = int(
                    input(
                        f"Enter the row (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                start_pos = Position(
                    x=start_col, y=abs(start_row - Hnefatafl.DIMENSION)
                )

                print("Enter the end position of the piece you wish to move:")
                end_col = input(
                    f"Enter the column (A - G) to play for the player {self.to_play()}: "
                )
                if "A" <= end_col <= "G":
                    end_col = ord(end_col) - ord("A")  # Convert letter to 0-based index
                else:
                    print("Invalid input! Please enter a letter between A and G.")
                    iter
                end_row = int(
                    input(
                        f"Enter the row (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                end_pos = Position(x=end_col, y=abs(end_row - Hnefatafl.DIMENSION))

                if not (
                    # action in self.legal_actions() and
                    0 <= start_pos.x
                    and 0 <= start_pos.y
                    and start_pos.x < Hnefatafl.DIMENSION
                    and start_pos.y < Hnefatafl.DIMENSION
                    and 0 <= end_pos.x
                    and 0 <= end_pos.y
                    and end_pos.x < Hnefatafl.DIMENSION
                    and end_pos.y < Hnefatafl.DIMENSION
                ):
                    print("Wrong move buddy")
                else:
                    action = self.env.move_to_action((start_pos, end_pos))
                    break
            except Exception as e:
                print("Error kp")
                print(e)
                pass
            print("Wrong input, try again")
        return action

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        start_pos, end_pos = self.env.action_to_move(action_number)
        return f"{start_pos.to_string()} -> {end_pos.to_string()}"


class GameResult(Enum):
    WIN = "win"
    ONGOING = "ongoing"
    DRAW = "draw"


class PlayerRole(Enum):
    DEFENDER = 1
    ATTACKER = -1

    def to_string(self) -> str:
        if self == PlayerRole.DEFENDER:
            return "Defender"
        return "Attacker"

    def toggle(self):
        return PlayerRole(self.value * -1)


class Hnefatafl:
    DIMENSION = 7
    INVALID_ACTION_REWARD = -10
    VALID_ACTION_REWARD = 10
    CAPTURE_REWARD = 5
    CAPTURED_REWARD = -5
    WIN_REWARD = 100
    LOSS_REWARD = -100

    DEFAULT_BOARD: List[List[Optional[PieceType]]] = [
        [
            None,  # Position(0, 0)
            None,  # Position(0, 1)
            None,  # Position(0, 2)
            PieceType.ATTACKER,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            PieceType.ATTACKER,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            PieceType.DEFENDER,
            None,
            None,
            None,
        ],
        [
            PieceType.ATTACKER,
            PieceType.ATTACKER,
            PieceType.DEFENDER,
            PieceType.KING,
            PieceType.DEFENDER,
            PieceType.ATTACKER,
            PieceType.ATTACKER,
        ],
        [
            None,
            None,
            None,
            PieceType.DEFENDER,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            PieceType.ATTACKER,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            PieceType.ATTACKER,
            None,
            None,
            None,
        ],
    ]

    def __init__(self, role=PlayerRole.ATTACKER):
        self.board = Hnefatafl.DEFAULT_BOARD.copy()
        self.player_role = role
        self.current_player = PlayerRole.ATTACKER
        self.king = Piece(PieceType.KING, Position(3, 3))
        self.attackers: List[Position] = self.__get_attackers()
        self.defenders: List[Position] = self.__get_defenders()

    def __get_attackers(self):
        attackers = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos = Position(i, j)
                if pos.get_square(self.board) == PieceType.ATTACKER:
                    attackers.append(pos)
        return attackers

    def __get_defenders(self):
        defenders = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos = Position(i, j)
                if pos.get_square(self.board) == PieceType.DEFENDER:
                    defenders.append(pos)
        return defenders

    def to_play(self):
        return 0 if self.player_role == 1 else 1

    def reset(self):
        self.board = Hnefatafl.DEFAULT_BOARD.copy()
        self.player = PlayerRole.ATTACKER
        return self.get_observation()

    def get_observation(self):
        """
        Returns the current observation.
        """
        observation = numpy.zeros((3, Hnefatafl.DIMENSION, Hnefatafl.DIMENSION))
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos = Position(i, j)
                if pos.get_square(self.board) == PieceType.ATTACKER:
                    observation[0, i, j] = 1
                elif pos.get_square(self.board) == PieceType.DEFENDER:
                    observation[1, i, j] = 1
                elif pos.get_square(self.board) == PieceType.KING:
                    observation[2, i, j] = 1
                """if self.board[i][j] == PieceType.ATTACKER:
                    observation[0, i, j] = 1
                elif self.board[i][j] == PieceType.DEFENDER:
                    observation[1, i, j] = 1
                elif self.board[i][j] == PieceType.KING:
                    observation[2, i, j] = 1"""
        return observation

    def get_possible_dests_from_pos(self, start_pos: Position) -> List[Position]:
        dests: List[Position] = []

        piece_to_move = start_pos.get_square(self.board)
        if piece_to_move is None:
            return dests

        if self.square_belongs_to_current_player(start_pos.get_square(self.board)):
            # move left
            k = 1
            while (
                start_pos.left(k).is_open_to_piece(piece_to_move)
                and start_pos.left(k).get_square(self.board) is None
            ):
                print(f"added {start_pos.left(k).to_string()}")
                dests.append(start_pos.left(k))
                k += 1
            # move right
            k = 1
            while (
                start_pos.right(k).is_open_to_piece(piece_to_move)
                and start_pos.right(k).get_square(self.board) is None
            ):
                dests.append(start_pos.right(k))
                k += 1
            # move up
            k = 1
            while (
                start_pos.up(k).is_open_to_piece(piece_to_move)
                and start_pos.up(k).get_square(self.board) is None
            ):
                dests.append(start_pos.up(k))
                k += 1
            # move down
            k = 1
            while (
                start_pos.down(k).is_open_to_piece(piece_to_move)
                and start_pos.down(k).get_square(self.board) is None
            ):
                dests.append(start_pos.down(k))
                k += 1
        return dests

    def get_possible_moves(self) -> List[Move]:
        """
        Returns a list of possible moves for the current player.
        """
        moves: List[Move] = []  # start_pos, end_pos
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                possible_moves_from_pos = self.get_possible_dests_from_pos(
                    Position(i, j)
                )
                for end_pos in possible_moves_from_pos:
                    moves.append((Position(i, j), end_pos))
        return moves

    def square_belongs_to_current_player(self, piece: Optional[PieceType]) -> bool:
        if piece is None:
            return False
        if piece == PieceType.ATTACKER:
            return self.current_player == PlayerRole.ATTACKER
        return self.current_player == PlayerRole.DEFENDER

    def game_over(self) -> Optional[Tuple[GameResult, PlayerRole]]:
        """
        Returns the game result along with the player if game was won.
        If GameResult is ONGOING, the second value is None.
        If GameResult is DRAW, the second value is None.
        """
        # The King Escapes
        if self.king.position in Position.CORNERS:
            print("king escaped")
            return GameResult.WIN, PlayerRole.DEFENDER
        # The King is Captured
        if self.king.captured:
            print("king captured")
            return GameResult.WIN, PlayerRole.ATTACKER
        # No Legal Moves
        if len(self.get_possible_moves()) == 0:
            print("no legal moves left")
            return GameResult.DRAW, None
        # All Defenders Are Eliminated
        if len(self.defenders) == 0:
            print("all defenders eleminated")
            return GameResult.WIN, PlayerRole.ATTACKER
        return GameResult.ONGOING, None

    def move_to_action(self, move: Move) -> int:
        """
        Encode the action as an integer.
        """
        start_pos = move[0]
        end_pos = move[1]
        _from = start_pos.x * Hnefatafl.DIMENSION + start_pos.y
        _to = end_pos.x * Hnefatafl.DIMENSION + end_pos.y
        return _from * Hnefatafl.DIMENSION**2 + _to

    def action_to_move(self, action: int) -> Move:
        """
        Decode the action from an integer into start_pos and end_pos.
        """
        _from, _to = action // Hnefatafl.DIMENSION**2, action % Hnefatafl.DIMENSION**2
        x0, y0 = _from // Hnefatafl.DIMENSION, _from % Hnefatafl.DIMENSION
        x1, y1 = _to // Hnefatafl.DIMENSION, _to % Hnefatafl.DIMENSION
        return (Position(x0, y0), Position(x1, y1))

    def step(self, action):
        reward = 0
        # check if action is legal
        start_pos, end_pos = self.action_to_move(action)
        if not self.belongs_to_me(start_pos.get_square(self.board)):
            print("Can't move from here")
            reward += Hnefatafl.INVALID_ACTION_REWARD
            return self.board, reward, False
        possible_dests = self.get_possible_dests_from_pos(start_pos)
        if len(possible_dests) == 0:
            # TODO: check other start positions (maybe can make moves from other start pos)
            # current_player loses
            print("No possible moves")
            reward += Hnefatafl.LOSS_REWARD
            return self.board, reward, True
        if end_pos not in possible_dests:
            print(f"Invalid move ({start_pos.to_string()} -> {end_pos.to_string()})")
            reward += Hnefatafl.INVALID_ACTION_REWARD
            return self.board, reward, False

        reward += Hnefatafl.VALID_ACTION_REWARD
        piece = start_pos.get_square(self.board)
        start_pos.set_square(self.board, None)
        end_pos.set_square(self.board, piece)
        if piece == PieceType.KING:
            self.king.position = end_pos
        print(
            f"Player {self.current_player} moved {piece.to_string()} from {start_pos.to_string()} to {end_pos.to_string()}"
        )

        # TODO: empty restricted piece can also capture opponent
        # check if moved piece captures an opponent
        # 1. up
        if (
            end_pos.y <= Hnefatafl.DIMENSION - 3
            and self.is_opponent(end_pos.up().get_square(self.board))
            and not self.king.position
            == end_pos.up()  # king can't be captured by two attackers
            and self.belongs_to_me(end_pos.up(2).get_square(self.board))
        ):
            # piece above captured
            print("Piece above captured")
            end_pos.up().set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD
        # 2. down
        if (
            end_pos.y >= 2
            and self.is_opponent(end_pos.down().get_square(self.board))
            and not self.king.position
            == end_pos.down()  # king can't be captured by two attackers
            and self.belongs_to_me(end_pos.down(2).get_square(self.board))
        ):
            # piece below captured
            print("Piece below captured")
            end_pos.down().set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD
        # 3. left
        if (
            end_pos.x >= 2
            and self.is_opponent(end_pos.left().get_square(self.board))
            and not self.king.position
            == end_pos.left()  # king can't be captured by two attackers
            and self.belongs_to_me(end_pos.left(2).get_square(self.board))
        ):
            # piece to left captured
            print("Piece to left captured")
            end_pos.left().set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD
        # 4. right
        if (
            end_pos.x <= Hnefatafl.DIMENSION - 3
            and self.is_opponent(end_pos.right().get_square(self.board))
            and not self.king.position
            == end_pos.right()  # king can't be captured by two attackers
            and self.belongs_to_me(end_pos.right(2).get_square(self.board))
        ):
            # piece to right captured
            print("Piece to right captured")
            end_pos.right().set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD

        """
        # check if moved piece itself is captured
        # 1. horizontally
        if end_pos.x >= 1 and end_pos.x <= Hnefatafl.DIMENSION - 2:
            if self.is_opponent(end_pos.right().get_square(self.board)) and self.is_opponent(end_pos.left().get_square(self.board)):
                print("Piece itself captured")
                end_pos.set_square(self.board, None)
                reward += Hnefatafl.CAPTURED_REWARD
        # 2. vertically
        if end_pos.y >= 1 and end_pos.y <= Hnefatafl.DIMENSION - 2:
            if self.is_opponent(end_pos.up().get_square(self.board)) and self.is_opponent(end_pos.down().get_square(self.board)):
                print("Piece itself captured")
                end_pos.set_square(self.board, None)
                reward += Hnefatafl.CAPTURED_REWARD
        """

        # check if moved piece captures the king
        # 1. king is on the throne
        if self.king.position == Position.MIDDLE:
            if (
                Position.MIDDLE.up().get_square(self.board) == PieceType.ATTACKER
                and Position.MIDDLE.left().get_square(self.board) == PieceType.ATTACKER
                and Position.MIDDLE.right().get_square(self.board) == PieceType.ATTACKER
                and Position.MIDDLE.down().get_square(self.board) == PieceType.ATTACKER
            ):
                print("King captured")
                self.king.captured = True
                Position.MIDDLE.set_square(self.board, None)
        # 2. king is next to the throne
        # 2.1 above
        if self.king.position == Position.MIDDLE.up():
            if (
                self.king.position.up().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.left().get_square(self.board)
                == PieceType.ATTACKER
                and self.king.position.right().get_square(self.board)
                == PieceType.ATTACKER
            ):
                print("King captured")
                self.king.position.set_square(self.board, None)
                self.king.captured = True
        # 2.2 below
        if self.king.position == Position.MIDDLE.down():
            if (
                self.king.position.down().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.left().get_square(self.board)
                == PieceType.ATTACKER
                and self.king.position.right().get_square(self.board)
                == PieceType.ATTACKER
            ):
                print("King captured")
                self.king.position.set_square(self.board, None)
                self.king.captured = True
        # 2.3 to the left
        if self.king.position == Position.MIDDLE.left():
            if (
                self.king.position.left().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.up().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.down().get_square(self.board)
                == PieceType.ATTACKER
            ):
                print("King captured")
                self.king.position.set_square(self.board, None)
                self.king.captured = True
        # 2.4 to the right
        if self.king.position == Position.MIDDLE.right():
            if (
                self.king.position.right().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.up().get_square(self.board) == PieceType.ATTACKER
                and self.king.position.down().get_square(self.board)
                == PieceType.ATTACKER
            ):
                print("King captured")
                self.king.position.set_square(self.board, None)
                self.king.captured = True

        # check if the game is over
        game_result, player = self.game_over()
        done = game_result != GameResult.ONGOING
        if game_result == GameResult.WIN:
            if player == self.player_role:
                reward += Hnefatafl.WIN_REWARD
            else:
                reward += Hnefatafl.LOSS_REWARD
        self.current_player = self.current_player.toggle()

        self.attackers = self.__get_attackers()
        self.defenders = self.__get_defenders()

        print(f"Done: {done}")

        return self.board, reward, done

    def is_opponent(self, piece_type: PieceType) -> bool:
        if piece_type is None:
            return False

        if self.current_player == PlayerRole.DEFENDER:
            return piece_type == PieceType.ATTACKER
        return piece_type == PieceType.DEFENDER or piece_type == PieceType.KING

    def belongs_to_me(self, piece_type: PieceType) -> bool:
        if piece_type is None:
            return False

        if self.current_player == PlayerRole.ATTACKER:
            return piece_type == PieceType.ATTACKER
        return piece_type == PieceType.DEFENDER or piece_type == PieceType.KING

    def render(self):
        # Print column numbers as the header
        col_numbers = "  " + "   ".join(
            f"{col}" for col in list(string.ascii_uppercase[: Hnefatafl.DIMENSION])
        )
        print(col_numbers)

        for i in range(Hnefatafl.DIMENSION):
            print("+---" * Hnefatafl.DIMENSION + "+")  # Top border of the row
            row_str = "|"
            for j in range(Hnefatafl.DIMENSION):
                cell = (
                    self.board[i][j].to_string() if self.board[i][j] else " "
                )  # Use piece or empty space
                row_str += f" {cell} |"
                if j == Hnefatafl.DIMENSION - 1:
                    row_str += f" {abs(i - Hnefatafl.DIMENSION)}"
            print(row_str)
        print("+---" * Hnefatafl.DIMENSION + "+")
