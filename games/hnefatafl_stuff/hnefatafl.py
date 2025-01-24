import copy
import string
from typing import List, Optional, Tuple

import numpy

from games.hnefatafl_stuff.direction import Direction
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
        default_board = copy.deepcopy(Hnefatafl.DEFAULT_BOARD)
        self.board = copy.deepcopy(default_board)
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
    def get_possible_moves(cls, board: Board, player: PlayerRole) -> List[Tuple[Position, int, int]]:
        """
        Returns a list of possible moves for the current player. excluding restricted positions
        """
        moves: List[Tuple[Position, int, int]] = []  # start_pos, direction, distance
        for i in range(Hnefatafl.DIMENSION):
            for j in range(Hnefatafl.DIMENSION):
                start_pos = Position(i, j)

                # skip restricted positions (corners) 
                if start_pos in [Hnefatafl.UPPER_LEFT, Hnefatafl.UPPER_RIGHT, Hnefatafl.LOWER_LEFT, Hnefatafl.LOWER_RIGHT]:
                    continue
                
                if Hnefatafl.piece_belongs_to_player(
                    piece=start_pos.get_square(board), player=player
                ):
                    possible_destinations = Hnefatafl.get_possible_dests_from_pos(
                        start_pos, board, player
                    )

                    for end_pos in possible_destinations:
                        direction, distance = cls.get_direction_and_distance(start_pos, end_pos)
                        moves.append((start_pos, direction, distance))
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
        # No Legal Moves -> handled in my_step
        """if len(Hnefatafl.get_possible_moves(board=board, player=player)) == 0:
            print("no legal moves left")
            return GameResult.DRAW, None"""
        # All Attackers Are Eliminated
        if len(attackers) == 0:
            print("all attackers eleminated")
            return GameResult.WIN, PlayerRole.DEFENDER
        return GameResult.ONGOING, None

   
    @classmethod
    def get_valid_start_positions(cls):
        """
        Returns a list of all valid start positions excluding restricted corners.
        """
        return [
            Position(x, y)
            for x in range(cls.DIMENSION)
            for y in range(cls.DIMENSION)
            if Position(x, y) not in cls.CORNERS
        ]

    def move_to_action(self, move: Move) -> int:
        """
        Encode the action as an integer.
        """
        start_pos = move[0]
        end_pos = move[1]
        
        valid_positions = Hnefatafl.get_valid_start_positions()

        _from = valid_positions.index(start_pos)
        direction, distance = self.get_direction_and_distance(start_pos, end_pos)

        action = _from * (4 * 6) + direction * 6 + (distance -1) #4 directions times move of size max 6
        return action

    def action_to_move(self, action: int) -> Move:
        """
        Decode the action from an integer into start_pos and end_pos.
        """
        valid_positions = Hnefatafl.get_valid_start_positions()

        # decode into start_index, direction and distance
        _from = action // (4 * 6)
        direction_distance = action % (4 *6)
        direction = direction_distance // 6
        distance = (direction_distance % 6) + 1

        
        start_pos = valid_positions[_from]

        # calculate end_pos based on direction and distance
        x0, y0 = start_pos.x, start_pos.y
        if direction == 0: # east
            end_pos = Position(x0 + distance, y0)
        elif direction == 1: # west
            end_pos = Position(x0 - distance, y0)
        elif direction == 2: # north
            end_pos = Position(x0, y0 + distance)
        elif direction == 3: #south
            end_pos = Position(x0, y0 - distance)
        else:
            raise ValueError("Invalid direction in action")

        return start_pos, end_pos

    @classmethod
    def get_direction_and_distance(cls, start_pos: Position, end_pos: Position) -> Tuple[int, int]:
        """
        Calculate the direction and distance between two positions.
        Directions are encoded as:
        0 - East, 1 - West, 2 - North, 3 - South
        """
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y

        if dx > 0 and dy == 0:
            return 0, abs(dx) # east
        elif dx < 0 and dy == 0:
            return 1, abs(dx) # west
        elif dy < 0 and dx == 0:
            return 2, abs(dy) # north
        elif dy > 0 and dx == 0:
            return 3, abs(dy) # south

        raise ValueError("Invalid move: Not a straight-line move.")

    @classmethod
    def piece_captured(
        cls,
        new_pos: Position,
        maybe_captured: Position,
        other_side: Position,
        player: PlayerRole,
        board: Board,
        king_pos: Position,
        capture_king: bool = False,
    ) -> bool:
        """
        Checks if a piece (only Attacker of Defender, NOT king) was captured.
        """
        if not new_pos.is_within_board():
            print("1")
            return False
        if not maybe_captured.is_within_board():
            print("2")
            return False
        if not other_side.is_within_board():
            print("3")
            return False
        if (
            maybe_captured.get_square(board=board) == PieceType.KING
            and not capture_king
        ):
            return False  # capturing of king is not handled here (but in king_captured)
        if not Hnefatafl.is_opponent(
            maybe_captured.get_square(board),
            of_player=player,
        ):
            print("4")
            return False

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
    def king_is_captured(
        cls,
        king_pos: Position,
        board: Board,
        new_pos: Position,  # new position of moved piece
        player: PlayerRole,
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

        # king is in "random" position (not on/next to throne, on restricted field)
        if (
            king_pos not in Hnefatafl.CORNERS
            and king_pos != Hnefatafl.MIDDLE
            and king_pos
            not in [
                Hnefatafl.MIDDLE.up(),
                Hnefatafl.MIDDLE.down(),
                Hnefatafl.MIDDLE.left(),
                Hnefatafl.MIDDLE.right(),
            ]
        ):
            adjacent_dir = new_pos.is_adjacent(king_pos)
            if adjacent_dir is not None:
                if Hnefatafl.piece_captured(
                    new_pos=new_pos,
                    maybe_captured=king_pos,
                    other_side=new_pos.get_adjacent_position(
                        direction=adjacent_dir, steps=2
                    ),
                    player=player,
                    board=board,
                    king_pos=king_pos,
                    capture_king=True,
                ):
                    return True
        return False

    @classmethod
    def my_step(
        cls,
        board: Board,
        move: Move,
        player: PlayerRole,
        king_pos: Position,
        attackers: List[Position],
    ) -> Tuple[
        bool,
        int,
        Board,
        Position,
        bool,
    ]:  # done, reward, updated_board, updated_king_pos, king_captured
        done = False
        reward = 0
        updated_king_pos = copy.deepcopy(king_pos)
        king_captured = False

        start_pos, end_pos = move

        # check if any moves left
        if len(Hnefatafl.get_possible_moves(board=board, player=player)) == 0:
            return True, Hnefatafl.LOSS_REWARD, board, king_pos

        # check if move is legal
        if not Hnefatafl.piece_belongs_to_player(
            piece=start_pos.get_square(board=board), player=player
        ):
            return False, Hnefatafl.INVALID_ACTION_REWARD, board, king_pos
        if not end_pos in Hnefatafl.get_possible_dests_from_pos(
            start_pos=start_pos, board=board, player=player
        ):
            return False, Hnefatafl.INVALID_ACTION_REWARD, board, king_pos

        # move piece from move[0] to move[1]
        reward += Hnefatafl.VALID_ACTION_REWARD
        end_pos.set_square(board=board, piece=start_pos.get_square(board=board))
        start_pos.set_square(board=board, piece=None)
        if end_pos.get_square(board) == PieceType.KING:
            updated_king_pos = end_pos

        # check if adjacent piece (to move[1]) was captured
        capture_res: Optional[Position] = None
        for direction in Direction:
            if Hnefatafl.piece_captured(
                new_pos=end_pos,
                maybe_captured=end_pos.get_adjacent_position(direction=direction),
                other_side=end_pos.get_adjacent_position(direction=direction, steps=2),
                player=player,
                board=board,
                king_pos=king_pos,
            ):
                capture_res = end_pos.get_adjacent_position(direction=direction)
        if not capture_res is None:
            capture_res.set_square(board, None)
            reward += Hnefatafl.CAPTURE_REWARD

        # check if king was captured
        if Hnefatafl.king_is_captured(
            king_pos=updated_king_pos,
            board=board,
            new_pos=end_pos,
            player=player,
        ):
            updated_king_pos.set_square(board=board, piece=None)
            updated_king_pos = Position(Position.INVALID, Position.INVALID)
            king_captured = True
            return True, reward, board, updated_king_pos, king_captured

        # check if game is over
        game_result, result_player = Hnefatafl.game_over(
            king_pos=updated_king_pos,
            king_captured=king_captured,
            board=board,
            player=player,
            attackers=attackers,
        )
        done = game_result != GameResult.ONGOING
        if game_result == GameResult.WIN:
            if result_player == player:
                reward += Hnefatafl.WIN_REWARD
            else:
                reward += Hnefatafl.LOSS_REWARD

        return done, reward, board, updated_king_pos, king_captured

    def step(self, action):
        # check if action is legal
        move = self.action_to_move(action)

        done, reward, updated_board, updated_king_pos, king_captured = (
            Hnefatafl.my_step(
                board=self.board,
                move=move,
                player=self.current_player,
                king_pos=self.king.position,
                attackers=self.attackers,
            )
        )
        self.board = updated_board
        self.king.position = updated_king_pos
        self.king.captured = king_captured

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
