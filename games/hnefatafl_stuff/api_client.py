import requests
from typing import List, Optional
from pydantic import BaseModel, Field

from games.hnefatafl_stuff.types import Board, Move
from games.hnefatafl_stuff.player_role import PlayerRole
from games.hnefatafl_stuff.position import Position


class Player(BaseModel):
    id: str
    name: str


class CreatePlayerRequest(BaseModel):
    name: str


class CreateGameRequest(BaseModel):
    player_type: PlayerRole = Field(alias="player_typer")


class CreateGameResponse(BaseModel):
    id: str


class JoinGameResponse(BaseModel):
    player_type: PlayerRole = Field(alias="type")


class GameState(BaseModel):
    board: Board
    current_player: PlayerRole


class MovesRequest(BaseModel):
    start_pos: Position


class MovePair(BaseModel):
    from_cell: List[Position] = Field(alias="from")
    to_cell: List[Position] = Field(alias="to")


class MovesResponse(BaseModel):
    moves: List[MovePair]


class MakeMoveRequest(BaseModel):
    player_id: str
    from_cell: List[Position] = Field(alias="from_cell")
    to_cell: List[Position] = Field(alias="to_cell")


class APIClient:
    _instance = None  # Singleton instance
    game_id: Optional[str]
    player_id: Optional[str]

    URL = "http://0.0.0.0:3020"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIClient, cls).__new__(cls)
        return cls._instance

    def create_player(self, name: str) -> Player:
        """Returns newly created player"""
        req = CreatePlayerRequest(name=name)
        response = requests.post(f"{APIClient.URL}/players", json=req)
        response.raise_for_status()  # Raises an exception if the request fails
        response_data = response.json()  # Parses the JSON response
        player = Player.model_validate(response_data)
        APIClient().player_id = player.id
        return player

    def create_game(self, player_role: PlayerRole) -> str:
        """Returns id of newly created game."""
        req = CreateGameRequest(player_type=player_role)
        response = requests.post(f"{APIClient.URL}/games", json=req)
        response.raise_for_status()
        response_data = response.json()
        create_game_response: CreateGameResponse = CreateGameResponse.model_validate(
            response_data
        )
        game_id = create_game_response.id
        APIClient().game_id = game_id
        return game_id

    def join_game(self, game_id: str) -> PlayerRole:
        """Return role of player [player_id] in newly joined game [game_id]"""
        assert self.player_id is not None
        assert game_id is not None or self.game_id is not None

        if game_id is not None:
            self.game_id = game_id

        response = requests.post(
            f"{APIClient.URL}/{self.player_id}/games/{self.game_id}",
            json=None,
        )
        response.raise_for_status()
        join_game_response: JoinGameResponse = JoinGameResponse.model_validate(
            response.json()
        )
        return join_game_response.player_type

    def get_game_state(self) -> GameState:
        assert self.game_id is not None

        response = requests.get(f"{APIClient.URL}/games/{self.game_id}")
        response.raise_for_status()
        game_state: GameState = GameState.model_validate(response.json())
        return game_state

    def get_possible_moves(self) -> MovesResponse:
        assert self.game_id is not None

        response = requests.get(f"{APIClient.URL}/games/{self.game_id}/moves")
        response.raise_for_status()
        moves_response: MovesResponse = MovesResponse.model_validate(response.json())
        return moves_response

    def make_move(self, move: Move):
        """
        Perform move [move] on player [player_id] in game [game_id].
        """
        assert self.game_id is not None

        move_req = MakeMoveRequest(
            player_id=self.player_id,
            from_cell=move[0].to_list(),
            to_cell=move[1].to_list(),
        )
        response = requests.post(
            f"{APIClient.URL}/games/{self.game_id}/move", json=move_req
        )
        response.raise_for_status()

    def wait_for_our_turn(self):
        """
        Check if you can move (blocking call).
        This will hang until the game logic determines it's your turn.
        """
        assert self.player_id is not None and self.game_id is not None

        response = requests.get(
            f"{APIClient.URL}/{self.player_id}/games/{self.game_id}/can_move"
        )
        response.raise_for_status()
