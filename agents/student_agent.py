"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES, state_key)
from constants import *
import numpy as np
import time
import math

#----------------------------------------------------------------------
EXACT = 0
BETA = 1
ALPHA = 2

class SearchTimeout(Exception):
    """Raised when the search approaches STEP_TIME_LIMIT."""
    pass
#-----------------------------------------------------------------------

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
       
        #----------------------------------------------------------------
        self.time_margin = 0.20
        self.deadline = 0.0
        self.nodes_visited = 0
        # state_key(board, reserves, active_player), [transpose_depth, transpose_flag,
        #                            transpose_score, transpose_move]
        self.transpose: dict[bytes, tuple[int, int, float, Move | None]] = {}
        #---------------------------------------------------------------

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        self.deadline = start + STEP_TIME_LIMIT - self.time_margin
        self.nodes_visited = 0
        self.transpose.clear()

        allow_moves = bool(ALLOW_MOVES)
        win_length = int(WIN_LENGTH)
        lines = LINES[win_length]

        valid_moves = get_valid_moves(board, reserves, player, allow_moves)
        if not valid_moves:
            return random_move(board, reserves, player, allow_moves=allow_moves)
        if len(valid_moves) == 1:
            return valid_moves[0]
      #—-------------------Part from Sofia------------------------ 
        for mv in valid_moves:
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv, allow_moves)
            if check_winner(b, player, win_length) == player:
                return mv
#----------------------------------------------------------------------------
        best_move = self._order_moves(board, valid_moves, opponent, None)[0]
       
        # 4X4 board size
        for depth in range(1, 16):
            if time.perf_counter() >= self.deadline:
                break
            try:
                score, move = self._alphabeta(
                    board=board,
                    reserves=reserves,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing=True,
                    player=player,
                    opponent=opponent,
                    last_mover=opponent,
                    allow_moves=allow_moves,
                    win_length=win_length,
                    lines=lines
                )
                if move is not None:
                    best_move = move
                if score >= 90000:
                    break
            except SearchTimeout:
                break

        return best_move

    def _alphabeta(self, board: np.ndarray, reserves: np.ndarray, depth: int,
                   alpha: float, beta: float, maximizing: bool,
                   player: int, opponent: int, last_mover: int,
                   allow_moves: bool, win_length: int, lines) -> tuple[float, Move | None]:
       
        self.nodes_visited += 1
        if (self.nodes_visited % 100) == 0:
            if time.perf_counter() >= self.deadline:
                raise SearchTimeout()
           
        winner = check_winner(board, last_mover, win_length)
        if winner == player:
            return 100000 + depth, None
        elif winner == opponent:
            return -100000 - depth, None

        active_player = player if maximizing else opponent
        initial_alpha = alpha

        key = state_key(board, reserves, active_player)
        transpose_move = None
        transpose_entry = self.transpose.get(key)
        if transpose_entry is not None:
            transpose_depth, transpose_flag, transpose_score, transpose_move = transpose_entry
            if transpose_depth >= depth:
                if transpose_flag == EXACT:
                    return transpose_score, transpose_move
                elif transpose_flag == BETA:
                    alpha = max(alpha, transpose_score)
                elif transpose_flag == ALPHA:
                    beta = min(beta, transpose_score)
                if alpha >= beta:
                    return transpose_score, transpose_move

        if depth == 0:
            return self._evaluate(board, reserves, player, opponent, win_length, lines), None

        valid_moves = get_valid_moves(board, reserves, active_player, allow_moves)
        if not valid_moves:
            if allow_moves:
                return (-100000 - depth if maximizing else 100000 + depth), None
            else:
                opponent_moves = get_valid_moves(board, reserves, 3 - active_player, allow_moves)
                if not opponent_moves:
                    return 0, None  
                score, _ = self._alphabeta(
                    board, reserves, depth - 1, alpha, beta,
                    not maximizing, player, opponent, active_player,
                    allow_moves, win_length, lines
                )
                return score, None

        other_player = opponent if active_player == player else player
        ordered_moves = self._order_moves(board, valid_moves, other_player, transpose_move)
        best_move = ordered_moves[0]

        if maximizing:
            max_eval = -math.inf
            for move in ordered_moves:
                next_board, next_reserves = copy_state(board, reserves)
                execute_move(next_board, next_reserves, active_player, move, allow_moves)

                eval_score, _ = self._alphabeta(
                    next_board, next_reserves, depth - 1, alpha, beta,
                    False, player, opponent, active_player,
                    allow_moves, win_length, lines
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            best_score = max_eval
        else:
            min_eval = math.inf
            for move in ordered_moves:
                next_board, next_reserves = copy_state(board, reserves)
                execute_move(next_board, next_reserves, active_player, move, allow_moves)

                eval_score, _ = self._alphabeta(
                    next_board, next_reserves, depth - 1, alpha, beta,
                    True, player, opponent, active_player,
                    allow_moves, win_length, lines
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            best_score = min_eval

        if best_score <= initial_alpha:
            flag = ALPHA
        elif best_score >= beta:
            flag = BETA
        else:
            flag = EXACT
        self.transpose[key] = (depth, flag, best_score, best_move)

        return best_score, best_move

    def _order_moves(self, board: np.ndarray, moves: list[Move],
                     other_player: int, transpose_move: Move | None) -> list[Move]:
        owners, sizes = top_view(board)
        own_list = owners.tolist()
        size_list = sizes.tolist()
        size_weights = (2, 5, 10)

        def priority(m: Move) -> int:
            if transpose_move is not None and m == transpose_move:
                return 10000
            score = m.size * 10
            r, c = m.dest

            # Reward gobbling an opponent piece, weighted by the size of the covered piece
            if own_list[r][c] == other_player:
                target_size = size_list[r][c]
                score += 25 + size_weights[target_size]

            if r in (1, 2) and c in (1, 2):
                score += 15
            elif r in (0, BOARD_SIZE - 1) and c in (0, BOARD_SIZE - 1):
                score += 8

            if m.src is None:
                score += 6
            return score

        return sorted(moves, key=priority, reverse=True)

    def _evaluate(self, board: np.ndarray, reserves: np.ndarray,
                  player: int, opponent: int, win_length: int, lines) -> float:
        owners, sizes = top_view(board)
        own_list = owners.tolist()
        size_list = sizes.tolist()
        score = 0.0
        threat_len = win_length - 1

        for line in lines:
            player_count = 0
            opponent_count = 0
            player_large = 0
            opponent_large = 0
            blocked = False

            for r, c in line:
                o = own_list[r][c]
                if o == OBSTACLE:
                    blocked = True
                    break
                elif o == player:
                    player_count += 1
                    if size_list[r][c] == LARGE:
                        player_large += 1
                elif o == opponent:
                    opponent_count += 1
                    if size_list[r][c] == LARGE:
                        opponent_large += 1
            if blocked:
                continue

            if player_count > 0 and opponent_count == 0:
                score += (14 ** player_count) + (player_large * 5)
                if player_count == threat_len:
                    score += 40.0
            elif opponent_count > 0 and player_count == 0:
                score -= (15 ** opponent_count) + (opponent_large * 6)
                if opponent_count == threat_len:
                    score -= 50.0

        size_weights = (2, 5, 10)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                top_owner = own_list[r][c]
                if top_owner == OBSTACLE:
                    continue

                if r in (1, 2) and c in (1, 2):
                    position_weight = 1.5
                    score += 15.0 if top_owner == player else (-15.0 if top_owner == opponent else 0.0)
                elif r in (0, BOARD_SIZE - 1) and c in (0, BOARD_SIZE - 1):
                    position_weight = 1.2
                else:
                    position_weight = 1.0

                top_size = size_list[r][c]
                if top_owner == player:
                    score += size_weights[top_size] * position_weight
                elif top_owner == opponent:
                    score -= size_weights[top_size] * position_weight

                stack = board[r, c]
                for s in range(top_size):
                    under_owner = stack[s]
                    if under_owner == player:
                        score += size_weights[s] * 0.3
                    elif under_owner == opponent:
                        score -= size_weights[s] * 0.3

        for s in (SMALL, MEDIUM, LARGE):
            score += (int(reserves[player, s]) - int(reserves[opponent, s])) * (size_weights[s] * 0.45)

        return score