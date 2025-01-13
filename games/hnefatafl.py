import datetime
from enum import Enum
import pathlib
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

class Position:
    """
    Represents a zero-based position on the board.
    """

    CAPTURED = -1

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def up(self) -> "Position":
        return Position(self.x + 1, self.y)

    def left(self) -> "Position":
        return Position(self.x, self.y - 1)

    def right(self) -> "Position":
        return Position(self.x, self.y + 1)

    def down(self) -> "Position":
        return Position(self.x - 1, self.y)


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
                print("Enter the start position of the piece you wish to move:")
                start_row = int(
                    input(
                        f"Enter the row (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                start_col = int(
                    input(
                        f"Enter the column (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                start_pos = Position(start_row - 1, start_col - 1)
                print("Enter the end position of the piece you wish to move:")
                end_row = int(
                    input(
                        f"Enter the row (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                end_col = int(
                    input(
                        f"Enter the column (1 - 7) to play for the player {self.to_play()}: "
                    )
                )
                end_pos = Position(end_row - 1, end_col - 1)
                
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
                    action = self.env.move_to_action(start_pos, end_pos)
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
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class GameResult(Enum):
    WIN = "win"
    ONGOING = "ongoing"
    DRAW = "draw"


class PlayerRole(Enum):
    DEFENDER = 1
    ATTACKER = -1


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
            return "🤴"

class Piece:
    def __init__(self, piece_type: PieceType, position: Position):
        self.piece_type = piece_type
        self.position = position
        self.captured = False


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
            None
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
        self.king = Position(3, 3)

    """
    def __get_attackers(self, board):
        attackers = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                if board[i, j] == PieceType.ATTACKER:
                    attackers.append(board[i, j])
        return attackers

    def __get_defenders(self, board):
        defenders = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                if board[i, j] == PieceType.DEFENDER:
                    defenders.append(board[i, j])
        return defenders
    """

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
                if self.board[i][j] == PieceType.ATTACKER:
                    observation[0, i, j] = 1
                elif self.board[i][j] == PieceType.DEFENDER:
                    observation[1, i, j] = 1
                elif self.board[i][j] == PieceType.KING:
                    observation[2, i, j] = 1
        return observation

    def get_possible_moves(self):
        """
        Returns a list of possible moves for the current player.
        """
        moves: List[Tuple[Position, Position]] = []  # start_pos, end_pos
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                if self.square_belongs_to_current_player(self.board[i][j]):
                    # move left
                    k = 1
                    while j - k >= 0 and self.board[i][j - k] is None:
                        moves.append((Position(i, j), Position(i, j - k)))
                        k += 1
                    # move right
                    k = 1
                    while j + k < Hnefatafl.DIMENSION and self.board[i][j + k] is None:
                        moves.append((Position(i, j), Position(i, j + k)))
                        k += 1
                    # move up
                    k = 1
                    while i + k < Hnefatafl.DIMENSION and self.board[i + k][j] is None:
                        moves.append((Position(i, j), Position(i + k, j)))
                        k += 1
                    # move down
                    k = 1
                    while i - k >= 0 and  self.board[i - k][j] is None:
                        moves.append((Position(i, j), Position(i - k, j)))
                        k += 1
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
        if self.king.position.x == 0 or self.king.position.x == Hnefatafl.DIMENSION - 1:
            return GameResult.WIN, PlayerRole.DEFENDER
        if self.king.position.y == 0 or self.king.position.y == Hnefatafl.DIMENSION - 1:
            return GameResult.WIN, PlayerRole.DEFENDER
        # The King is Captured
        if self.king.captured:
            return GameResult.WIN, PlayerRole.ATTACKER
        # No Legal Moves
        if len(self.get_possible_moves()) == 0:
            return GameResult.DRAW, None
        # All Defenders Are Eliminated
        for defender in self.defenders:
            if not defender.captured:
                return GameResult.ONGOING, None
            return GameResult.WIN, PlayerRole.ATTACKER
        return GameResult.ONGOING, None

    def move_to_action(
        self, start_pos: Position, end_pos: Position) -> int:
        """
        Encode the action as an integer.
        """
        _from = start_pos.x * Hnefatafl.DIMENSION + start_pos.y
        _to = end_pos.x * Hnefatafl.DIMENSION + end_pos.y
        return _from * Hnefatafl.DIMENSION**2 + _to

    def action_to_move(self, action: int) -> Tuple[Position, Position]:
        """
        Decode the action from an integer into start_pos and end_pos.
        """
        _from, _to = action // Hnefatafl.DIMENSION**2, action % Hnefatafl.DIMENSION**2
        x0, y0 = _from // Hnefatafl.DIMENSION, _from % Hnefatafl.DIMENSION
        x1, y1 = _to // Hnefatafl.DIMENSION, _to % Hnefatafl.DIMENSION
        return Position(x0, y0), Position(x1, y1)

    def step(self, action):
        reward = 0
        # check if action is legal
        move = self.action_to_move(action)
        possible_moves = self.get_possible_moves()
        if possible_moves.count == 0:
            # current_player loses
            reward += Hnefatafl.LOSS_REWARD
            return self.board, reward, True
        if move not in possible_moves:
            reward += Hnefatafl.INVALID_ACTION_REWARD
            return self.board, reward, False
        start_pos, end_pos = move
        self.board[start_pos] = None
        piece = self.board[start_pos]
        piece.position = end_pos
        self.board[end_pos] = piece

        # check if moved piece captures an opponent
        # 1. up
        if (
            end_pos.y <= Hnefatafl.DIMENSION - 3
            and self.is_opponent(self.board[end_pos.up()])
            and not self.is_opponent(self.board[end_pos.up().up()])
        ):
            # piece above captured
            # self.board[end_pos.up()].captured = True
            self.board[end_pos.up()] = None
            reward += Hnefatafl.CAPTURE_REWARD
        # 2. down
        if (
            end_pos.y >= 2
            and self.is_opponent(self.board[end_pos.down()])
            and not self.is_opponent(self.board[end_pos.down().down()])
        ):
            # piece below captured
            # self.board[end_pos.down()].captured = True
            self.board[end_pos.down()] = None
            reward += Hnefatafl.CAPTURE_REWARD
        # 3. left
        if (
            end_pos.x >= 2
            and self.is_opponent(self.board[end_pos.left()])
            and not self.is_opponent(self.board[end_pos.left().left()])
        ):
            # piece to left captured
            #vself.board[end_pos.left()].captured = True
            self.board[end_pos.left()] = None
            reward += Hnefatafl.CAPTURE_REWARD
        # 4. right
        if (
            end_pos.x <= Hnefatafl.DIMENSION - 3
            and self.is_opponent(self.board[end_pos.right()])
            and not self.is_opponent(self.board[end_pos.right().right()])
        ):
            # piece to right captured
            self.board[end_pos.right()].captured = True
            self.board[end_pos.right()] = None
            reward += Hnefatafl.CAPTURE_REWARD

        # check if moved piece itself is captured
        # 1. horizontally
        if end_pos.x >= 1 and end_pos.x <= Hnefatafl.DIMENSION - 2:
            if self.is_opponent(self.board[end_pos.right()]) and self.is_opponent(
                self.board[end_pos.left()]
            ):
                self.board[end_pos].captured = True
                self.board[end_pos] = None
                reward += Hnefatafl.CAPTURED_REWARD
        # 2. vertically
        if end_pos.y >= 1 and end_pos.y <= Hnefatafl.DIMENSION - 2:
            if self.is_opponent(self.board[end_pos.up()]) and self.is_opponent(
                self.board[end_pos.down()]
            ):
                self.board[end_pos].captured = True
                self.board[end_pos] = None
                reward += Hnefatafl.CAPTURED_REWARD

        # check if moved piece captures the king
        # 1. king is on the throne
        if self.king.position == Position(3, 3):
            if (
                self.board[2, 3] == PieceType.ATTACKER
                and self.board[3, 2] == PieceType.ATTACKER
                and self.board[4, 3] == PieceType.ATTACKER
                and self.board[3, 4] == PieceType.ATTACKER
            ):
                self.king.captured = True
                self.board[3, 3] = None
        # 2. king is next to the throne
        # 2.1 above
        if self.king.position == Position(3, 3).up():
            if (
                self.board[self.king.position.up()] == PieceType.ATTACKER
                and self.board[self.king.position.left()] == PieceType.ATTACKER
                and self.board[self.king.position.right()] == PieceType.ATTACKER
            ):
                self.king.captured = True
                self.board[3, 3] = None
        # 2.2 below
        if self.king.position == Position(3, 3).down():
            if (
                self.board[self.king.position.down()] == PieceType.ATTACKER
                and self.board[self.king.position.left()] == PieceType.ATTACKER
                and self.board[self.king.position.right()] == PieceType.ATTACKER
            ):
                self.king.captured = True
                self.board[3, 3] = None
        # 2.3 to the left
        if self.king.position == Position(3, 3).left():
            if (
                self.board[self.king.position.left()] == PieceType.ATTACKER
                and self.board[self.king.position.up()] == PieceType.ATTACKER
                and self.board[self.king.position.down()] == PieceType.ATTACKER
            ):
                self.king.captured = True
                self.board[3, 3] = None
        # 2.4 to the right
        if self.king.position == Position(3, 3).right():
            if (
                self.board[self.king.position.right()] == PieceType.ATTACKER
                and self.board[self.king.position.up()] == PieceType.ATTACKER
                and self.board[self.king.position.down()] == PieceType.ATTACKER
            ):
                self.king.captured = True
                self.board[3, 3] = None

        # check if the game is over
        game_result, player = self.game_over()
        done = game_result != GameResult.ONGOING
        if game_result == GameResult.WIN:
            if player == self.player_role:
                reward += Hnefatafl.WIN_REWARD
            else:
                reward += Hnefatafl.LOSS_REWARD
        self.current_player *= -1

        return self.board, reward, done

    def is_opponent(self, piece: Piece) -> bool:
        if self.current_player == PlayerRole.DEFENDER:
            return piece.piece_type == PieceType.ATTACKER
        return (
            piece.piece_type == PieceType.DEFENDER or piece.piece_type == PieceType.KING
        )

    def render(self):
        for i in range(Hnefatafl.DIMENSION):
            print("+---" * Hnefatafl.DIMENSION + "+")  # Top border of the row
            row_str = "|"
            for j in range(Hnefatafl.DIMENSION):
                cell = self.board[i][j].to_string() if self.board[i][j] else " "  # Use piece or empty space
                row_str += f" {cell} |"
            print(row_str)
        print("+---" * Hnefatafl.DIMENSION + "+")
