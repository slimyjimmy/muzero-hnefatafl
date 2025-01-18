import string
from typing import List, Optional, Tuple

import numpy

from games.hnefatafl_stuff.game_result import GameResult
from games.hnefatafl_stuff.piece import Piece
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position


Move = Tuple[Position, Position]


class Hnefatafl:
    DIMENSION = 7

    # Rewards
    INVALID_ACTION_REWARD = -10
    VALID_ACTION_REWARD = 10
    CAPTURE_REWARD = 5
    CAPTURED_REWARD = -5
    WIN_REWARD = 100
    LOSS_REWARD = -100

    # Positions
    MIDDLE = Position(3, 3)
    UPPER_LEFT = Position(0, 0)
    UPPER_RIGHT = Position(DIMENSION - 1, 0)
    LOWER_LEFT = Position(0, DIMENSION - 1)
    LOWER_RIGHT = Position(DIMENSION - 1, DIMENSION - 1)
    CORNERS = [UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT]
    RESTRICTED_POSITIONS = [UPPER_LEFT, UPPER_RIGHT, LOWER_LEFT, LOWER_RIGHT, MIDDLE]

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
        if self.king.position in Hnefatafl.CORNERS:
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

    def piece_captured(
        self, new_pos: Position, maybe_captured: Position, other_side: Position
    ) -> bool:
        if not new_pos.is_within_board():
            return False
        if not maybe_captured.is_within_board():
            return False
        if not other_side.is_within_board():
            return False
        if not self.is_opponent(maybe_captured.get_square(self.board)):
            return False
        if maybe_captured.get_square(self.board) == PieceType.KING:
            return False  # king can't be captured by two pieces

        # to capture maybe_captured, other_side can either be
        # - a piece belonging to current_player
        # - a hostile field
        #     - a corner
        #     - maybe_captured is defender -> empty throne
        #     - maybe_captured is attacker -> throne
        if self.belongs_to_me(other_side.get_square(self.board)):
            return True
        if other_side in Hnefatafl.CORNERS:
            return True
        if maybe_captured.get_square(self.board) == PieceType.DEFENDER:
            if (
                other_side == Hnefatafl.MIDDLE
                and self.king.position != Hnefatafl.MIDDLE
            ):
                return True
        if maybe_captured.get_square(self.board) == PieceType.ATTACKER:
            if other_side == Hnefatafl.MIDDLE:
                return True
        return False

    def piece_captures_opponent(self, end_pos: Position) -> Optional[Position]:
        """
        Returns position of captured opponent if the piece moved to its new position `end_pos`captures an opponent.
        Returns None otherwise.
        """
        if self.piece_captured(
            new_pos=end_pos, maybe_captured=end_pos.up(), other_side=end_pos.up(2)
        ):
            return end_pos.up()

        if self.piece_captured(
            new_pos=end_pos, maybe_captured=end_pos.down(), other_side=end_pos.down(2)
        ):
            return end_pos.down()

        if self.piece_captured(
            new_pos=end_pos, maybe_captured=end_pos.left(), other_side=end_pos.left(2)
        ):
            return end_pos.left()

        if self.piece_captured(
            new_pos=end_pos, maybe_captured=end_pos.right(), other_side=end_pos.right(2)
        ):
            return end_pos.right()

        return None

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

        capture_res = self.piece_captures_opponent(end_pos)
        if not capture_res is None:
            capture_res.set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD

        # check if moved piece captures the king
        if (
            Hnefatafl.MIDDLE.up().get_square(self.board) == PieceType.ATTACKER
            and Hnefatafl.MIDDLE.left().get_square(self.board) == PieceType.ATTACKER
            and Hnefatafl.MIDDLE.right().get_square(self.board) == PieceType.ATTACKER
            and Hnefatafl.MIDDLE.down().get_square(self.board) == PieceType.ATTACKER
        ):
            print("King captured")
            self.king.captured = True
            Hnefatafl.MIDDLE.set_square(self.board, None)
        # 2. king is next to the throne
        # 2.1 above
        if self.king.position == Hnefatafl.MIDDLE.up():
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
        if self.king.position == Hnefatafl.MIDDLE.down():
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
        if self.king.position == Hnefatafl.MIDDLE.left():
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
        if self.king.position == Hnefatafl.MIDDLE.right():
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
