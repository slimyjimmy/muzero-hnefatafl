import copy
import string
from typing import List, Optional, Tuple

import numpy

from games.hnefatafl_stuff.game_result import GameResult
from games.hnefatafl_stuff.piece import Piece
from games.hnefatafl_stuff.piece_type import PieceType
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position
from games.hnefatafl_stuff.types import Board, Move


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

    DEFAULT_BOARD: Board = [
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
        default_board = Hnefatafl.DEFAULT_BOARD.copy()
        self.board = default_board
        self.player_role = role
        self.current_player = PlayerRole.ATTACKER
        self.king = Piece(PieceType.KING, Position(3, 3))
        self.attackers: List[Position] = Hnefatafl.get_attackers(default_board)
        self.defenders: List[Position] = Hnefatafl.get_defenders(default_board)

    @classmethod
    def get_attackers(cls, board: Board) -> List[Position]:
        attackers: List[Position] = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos = Position(i, j)
                if pos.get_square(board) == PieceType.ATTACKER:
                    attackers.append(pos)
        return attackers

    @classmethod
    def get_defenders(cls, board: Board) -> List[Position]:
        defenders: List[Position] = []
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos = Position(i, j)
                if pos.get_square(board) == PieceType.DEFENDER:
                    defenders.append(pos)
        return defenders

    def reset(self):
        self.board = copy.deepcopy(Hnefatafl.DEFAULT_BOARD)
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
        return observation

    def to_play(self):
        return 0 if self.player_role == 1 else 1

    @classmethod
    def get_possible_dests_from_pos(
        cls,
        start_pos: Position,
        board: Board,
        player: PlayerRole,
    ) -> List[Position]:
        dests: List[Position] = []

        piece_to_move = start_pos.get_square(board)
        if piece_to_move is None:
            return dests

        if Hnefatafl.piece_belongs_to_player(
            piece=start_pos.get_square(board), player=player
        ):
            # move left
            k = 1
            while (
                start_pos.left(k).is_open_to_piece(piece_to_move)
                and start_pos.left(k).get_square(board) is None
            ):
                print(f"added {start_pos.left(k).to_string()}")
                dests.append(start_pos.left(k))
                k += 1
            # move right
            k = 1
            while (
                start_pos.right(k).is_open_to_piece(piece_to_move)
                and start_pos.right(k).get_square(board) is None
            ):
                dests.append(start_pos.right(k))
                k += 1
            # move up
            k = 1
            while (
                start_pos.up(k).is_open_to_piece(piece_to_move)
                and start_pos.up(k).get_square(board) is None
            ):
                dests.append(start_pos.up(k))
                k += 1
            # move down
            k = 1
            while (
                start_pos.down(k).is_open_to_piece(piece_to_move)
                and start_pos.down(k).get_square(board) is None
            ):
                dests.append(start_pos.down(k))
                k += 1
        return dests

    @classmethod
    def get_possible_moves(cls, board: Board, player: PlayerRole) -> List[Move]:
        """
        Returns a list of possible moves for the current player.
        """
        moves: List[Move] = []  # start_pos, end_pos
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                pos: Position = Position(i, j)
                if Hnefatafl.piece_belongs_to_player(
                    piece=pos.get_square(board), player=player
                ):
                    possible_moves_from_pos = Hnefatafl.get_possible_dests_from_pos(
                        pos, board, player
                    )
                    for end_pos in possible_moves_from_pos:
                        moves.append((pos, end_pos))
        return moves

    @classmethod
    def piece_belongs_to_player(
        cls,
        piece: Optional[PieceType],
        player: PlayerRole,
    ) -> bool:
        if piece is None:
            return False

        if player == PlayerRole.ATTACKER:
            return piece == PieceType.ATTACKER
        return piece == PieceType.DEFENDER or piece == PieceType.KING

    def belongs_to_me(self, piece_type: PieceType) -> bool:
        return Hnefatafl.piece_belongs_to_player(
            piece=piece_type, player=self.current_player
        )

    @classmethod
    def game_over(
        cls,
        king_pos: Optional[Position],
        king_captured: bool,
        board: Board,
        player: PlayerRole,
        attackers: List[Position],
    ) -> Optional[Tuple[GameResult, PlayerRole]]:
        """
        Returns the game result along with the player if game was won.
        If GameResult is ONGOING, the second value is None.
        If GameResult is DRAW, the second value is None.
        """

        if not king_captured and king_pos is None:
            raise ValueError("King was not captured, but no position given")

        if not king_captured and king_pos.get_square(board=board) != PieceType.KING:
            raise ValueError("King not in king_pos")

        # The King Escapes
        if king_pos in Hnefatafl.CORNERS:
            print("king escaped")
            return GameResult.WIN, PlayerRole.DEFENDER
        # The King is Captured
        if king_captured:
            print("king captured")
            return GameResult.WIN, PlayerRole.ATTACKER
        # No Legal Moves
        if len(Hnefatafl.get_possible_moves(board=board, player=player)) == 0:
            print("no legal moves left")
            return GameResult.DRAW, None
        # All Attackers Are Eliminated
        if len(attackers) == 0:
            print("all attackers eleminated")
            return GameResult.WIN, PlayerRole.DEFENDER
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

    @classmethod
    def piece_captured(
        cls,
        new_pos: Position,
        maybe_captured: Position,
        other_side: Position,
        player: PlayerRole,
        board: Board,
        king_pos: Position,
    ) -> bool:
        if not new_pos.is_within_board():
            print("1")
            return False
        if not maybe_captured.is_within_board():
            print("2")
            return False
        if not other_side.is_within_board():
            print("3")
            return False
        if not Hnefatafl.is_opponent(
            maybe_captured.get_square(board),
            of_player=player,
        ):
            print("4")
            return False
        if maybe_captured.get_square(board) == PieceType.KING:
            print("5")
            return False  # king can't be captured by two pieces

        # to capture maybe_captured, other_side can either be
        # - a piece belonging to current_player
        # - a hostile field
        #     - a corner
        #     - maybe_captured is defender -> empty throne
        #     - maybe_captured is attacker -> throne
        if Hnefatafl.piece_belongs_to_player(
            piece=other_side.get_square(board),
            player=player,
        ):
            return True
        if other_side in Hnefatafl.CORNERS:
            return True
        if maybe_captured.get_square(board) == PieceType.DEFENDER:
            if other_side == Hnefatafl.MIDDLE and king_pos != Hnefatafl.MIDDLE:
                return True
        if maybe_captured.get_square(board) == PieceType.ATTACKER:
            if other_side == Hnefatafl.MIDDLE:
                return True
        print("6")
        return False

    @classmethod
    def piece_captures_opponent(
        cls,
        end_pos: Position,
        board: Board,
        player: PlayerRole,
        king_pos: Position,
    ) -> Optional[Position]:
        """
        Returns position of captured opponent if the piece moved to its new position `end_pos`captures an opponent.
        Returns None otherwise.
        """
        if Hnefatafl.piece_captured(
            new_pos=end_pos,
            maybe_captured=end_pos.up(),
            other_side=end_pos.up(2),
            player=player,
            board=board,
            king_pos=king_pos,
        ):
            return end_pos.up()

        if Hnefatafl.piece_captured(
            new_pos=end_pos,
            maybe_captured=end_pos.down(),
            other_side=end_pos.down(2),
            player=player,
            board=board,
            king_pos=king_pos,
        ):
            return end_pos.down()

        if Hnefatafl.piece_captured(
            new_pos=end_pos,
            maybe_captured=end_pos.left(),
            other_side=end_pos.left(2),
            player=player,
            board=board,
            king_pos=king_pos,
        ):
            return end_pos.left()

        if Hnefatafl.piece_captured(
            new_pos=end_pos,
            maybe_captured=end_pos.right(),
            other_side=end_pos.right(2),
            player=player,
            board=board,
            king_pos=king_pos,
        ):
            return end_pos.right()

        return None

    @classmethod
    def king_is_captured(
        cls,
        king_pos: Position,
        board: Board,
    ) -> bool:

        if king_pos.get_square(board=board) != PieceType.KING:
            raise ValueError("King not in given king_pos on given board")

        # check if moved piece captures the king
        if (
            king_pos.up().get_square(board) == PieceType.ATTACKER
            and king_pos.left().get_square(board) == PieceType.ATTACKER
            and king_pos.right().get_square(board) == PieceType.ATTACKER
            and king_pos.down().get_square(board) == PieceType.ATTACKER
        ):
            return True
        # 2. king is next to the throne
        # 2.1 above
        if king_pos == Hnefatafl.MIDDLE.up():
            if (
                king_pos.up().get_square(board) == PieceType.ATTACKER
                and king_pos.left().get_square(board) == PieceType.ATTACKER
                and king_pos.right().get_square(board) == PieceType.ATTACKER
            ):
                return True
        # 2.2 below
        if king_pos == Hnefatafl.MIDDLE.down():
            if (
                king_pos.down().get_square(board) == PieceType.ATTACKER
                and king_pos.left().get_square(board) == PieceType.ATTACKER
                and king_pos.right().get_square(board) == PieceType.ATTACKER
            ):
                return True
        # 2.3 to the left
        if king_pos == Hnefatafl.MIDDLE.left():
            if (
                king_pos.left().get_square(board) == PieceType.ATTACKER
                and king_pos.up().get_square(board) == PieceType.ATTACKER
                and king_pos.down().get_square(board) == PieceType.ATTACKER
            ):
                return True
        # 2.4 to the right
        if king_pos == Hnefatafl.MIDDLE.right():
            if (
                king_pos.right().get_square(board) == PieceType.ATTACKER
                and king_pos.up().get_square(board) == PieceType.ATTACKER
                and king_pos.down().get_square(board) == PieceType.ATTACKER
            ):
                return True
        return False

    def step(self, action):
        reward = 0
        # check if action is legal
        start_pos, end_pos = self.action_to_move(action)
        if not self.belongs_to_me(start_pos.get_square(self.board)):
            print("Can't move from here")
            reward += Hnefatafl.INVALID_ACTION_REWARD
            return self.board, reward, False
        possible_dests = Hnefatafl.get_possible_dests_from_pos(
            start_pos=start_pos,
            board=self.board,
            player=self.current_player,
        )
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

        capture_res = Hnefatafl.piece_captures_opponent(
            end_pos=end_pos,
            board=self.board,
            player=self.current_player,
            king_pos=self.king.position,
        )
        if not capture_res is None:
            capture_res.set_square(self.board, None)
            reward += Hnefatafl.CAPTURE_REWARD

        # check if piece captures king
        if self.king_is_captured(king_pos=self.king.position, board=self.board):
            print("King captured")
            self.king.position.set_square(self.board, None)
            self.king.captured = True

        # check if the game is over
        game_result, player = self.game_over(
            king_pos=self.king.position,
            king_captured=self.king.captured,
            board=self.board,
            player=self.current_player,
            attackers=self.attackers,
        )
        done = game_result != GameResult.ONGOING
        if game_result == GameResult.WIN:
            if player == self.player_role:
                reward += Hnefatafl.WIN_REWARD
            else:
                reward += Hnefatafl.LOSS_REWARD
        self.current_player = self.current_player.toggle()

        self.attackers = Hnefatafl.get_attackers(self.board)
        self.defenders = Hnefatafl.get_defenders(self.board)

        return self.board, reward, done

    @classmethod
    def is_opponent(
        cls,
        piece_type: PieceType,
        of_player: PlayerRole,
    ) -> bool:
        if piece_type is None:
            return False

        if of_player == PlayerRole.DEFENDER:
            return piece_type == PieceType.ATTACKER
        return piece_type == PieceType.DEFENDER or piece_type == PieceType.KING

    def render(self) -> None:
        print(self.get_rendering_string(board=self.board))

    @classmethod
    def get_rendering_string(cls, board: Board) -> string:
        res = ""
        # Print column numbers as the header
        col_numbers = "  " + "   ".join(
            f"{col}" for col in list(string.ascii_uppercase[: Hnefatafl.DIMENSION])
        )
        res += col_numbers + "\n"

        for i in range(Hnefatafl.DIMENSION):
            res += "+---" * Hnefatafl.DIMENSION + "+" + "\n"  # Top border of the row
            row_str = "|"
            for j in range(Hnefatafl.DIMENSION):
                cell = (
                    board[i][j].to_string() if board[i][j] else " "
                )  # Use piece or empty space
                row_str += f" {cell} |"
                if j == Hnefatafl.DIMENSION - 1:
                    row_str += f" {abs(i - Hnefatafl.DIMENSION)}"
            res += row_str + "\n"
        res += "+---" * Hnefatafl.DIMENSION + "+" + "\n"
        return res
