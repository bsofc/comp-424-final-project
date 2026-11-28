"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (
    Move, get_valid_moves, execute_move, check_winner,
    check_endgame, random_move, copy_state, top_view, LINES, state_key
)
from constants import *
import numpy as np
import time
import math

EXACT = 0
BETA = 1
ALPHA = 2


class SearchTimeout(Exception):
    pass


@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.time_margin = 0.20
        self.deadline = 0.0
        self.nodes_visited = 0
        self.transpose = {}

    def step(self, board, reserves, player, opponent):

        start = time.perf_counter()
        self.deadline = start + STEP_TIME_LIMIT - self.time_margin
        self.nodes_visited = 0
        self.transpose.clear()

        allow_moves = bool(ALLOW_MOVES)
        win_length = int(WIN_LENGTH)
        lines = LINES[win_length]

        valid_moves = get_valid_moves(board, reserves, player, allow_moves)

        if not valid_moves:
            return random_move(board, reserves, player, allow_moves=allow_moves)

      
        for mv in valid_moves:
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv, allow_moves)
            if check_winner(b, player, win_length) == player:
                return mv

        if len(valid_moves) == 1:
            return valid_moves[0]

        best_move = self._order_moves(board, valid_moves, opponent, None)[0]

    
        for depth in range(1, 16):
            if time.perf_counter() >= self.deadline:
                break

            try:
                score, move = self._alphabeta(
                    board, reserves, depth,
                    -math.inf, math.inf,
                    True, player, opponent, opponent,
                    allow_moves, win_length, lines
                )

                if move is not None:
                    best_move = move

                if score >= 90000:
                    break

            except SearchTimeout:
                break

        return best_move

    def _alphabeta(self, board, reserves, depth,
                   alpha, beta, maximizing,
                   player, opponent, last_mover,
                   allow_moves, win_length, lines):

        self.nodes_visited += 1

        if self.nodes_visited % 100 == 0:
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
            return self.evaluate(board, reserves, player, opponent, win_length, lines), None

        valid_moves = get_valid_moves(board, reserves, active_player, allow_moves)

        if not valid_moves:
            if allow_moves:
                return (-100000 - depth if maximizing else 100000 + depth), None
            else:
                opponent_moves = get_valid_moves(
                    board, reserves, 3 - active_player, allow_moves
                )

                if not opponent_moves:
                    return 0, None

                score, _ = self._alphabeta(
                    board, reserves, depth - 1, alpha, beta,
                    not maximizing, player, opponent, active_player,
                    allow_moves, win_length, lines
                )
                return score, None

        other_player = opponent if active_player == player else player

        ordered_moves = self._order_moves(
            board, valid_moves, other_player, transpose_move
        )

        best_move = ordered_moves[0]

        if maximizing:
            max_eval = -math.inf

            for move in ordered_moves:
                next_board, next_reserves = copy_state(board, reserves)

                execute_move(
                    next_board, next_reserves,
                    active_player, move, allow_moves
                )

                eval_score, _ = self._alphabeta(
                    next_board, next_reserves, depth - 1,
                    alpha, beta, False,
                    player, opponent, active_player,
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

                execute_move(
                    next_board, next_reserves,
                    active_player, move, allow_moves
                )

                eval_score, _ = self._alphabeta(
                    next_board, next_reserves, depth - 1,
                    alpha, beta, True,
                    player, opponent, active_player,
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

        self.transpose[key] = (
            depth, flag, best_score, best_move
        )

        return best_score, best_move

    def _order_moves(self, board, moves, other_player, transpose_move):

        owners, sizes = top_view(board)
        own_list = owners.tolist()
        size_list = sizes.tolist()

        def priority(m):
            if transpose_move is not None and m == transpose_move:
                return 10000

            score = m.size * 10
            r, c = m.dest

            
            if own_list[r][c] == other_player:
                score += 75 + (2, 5, 10)[int(size_list[r][c])]

         
            if r in (1, 2) and c in (1, 2):
                score += 50

           
            elif r in (0, BOARD_SIZE - 1) and c in (0, BOARD_SIZE - 1):
                score += 30

            if m.src is None:
                score += 10

            return score

        return sorted(moves, key=priority, reverse=True)

    def evaluate(self, b, r, p, o, win_length, lines):

        score = 0.0

        owners, sizes = top_view(b)
        own_list = owners.tolist()
        size_list = sizes.tolist()

        
        for line in lines:

            player_count = 0
            opponent_count = 0
            player_large = 0
            opponent_large = 0
            blocked = False

            for row, col in line:

                owner = own_list[row][col]

                if owner == OBSTACLE:
                    blocked = True
                    break

                elif owner == p:
                    player_count += 1

                    if size_list[row][col] == LARGE:
                        player_large += 1

                elif owner == o:
                    opponent_count += 1

                    if size_list[row][col] == LARGE:
                        opponent_large += 1

            if blocked:
                continue

            if player_count > 0 and opponent_count == 0:
                score += (14 ** player_count) + (player_large * 5)

                if player_count == win_length - 1:
                    score += 40.0

            elif opponent_count > 0 and player_count == 0:
                score -= (15 ** opponent_count) + (opponent_large * 6)

                if opponent_count == win_length - 1:
                    score -= 50.0

       
        center_coords = {(1, 1), (1, 2), (2, 1), (2, 2)}

        corner_coords = {
            (0, 0), (0, BOARD_SIZE - 1),
            (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)
        }

        outer_corner_cells = {
            (0, 1), (0, 2), (1, 0), (1, 3),
            (2, 0), (2, 3), (3, 1), (3, 2)
        }

        size_weights = (2, 5, 10)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                owner = own_list[row][col]

                if owner == OBSTACLE:
                    continue

                if (row, col) in center_coords:
                    position_weight = 1.5
                    positional_bonus = 50
                elif (row, col) in corner_coords:
                    position_weight = 1.2
                    positional_bonus = 0
                elif (row, col) in outer_corner_cells:
                    position_weight = 1.0
                    positional_bonus = 0
                else:
                    position_weight = 1.0
                    positional_bonus = 0

                top_size = size_list[row][col]

                if owner == p:
                    score += size_weights[top_size] * position_weight
                    score += positional_bonus

                elif owner == o:
                    score -= size_weights[top_size] * position_weight
                    score -= positional_bonus

                stack = b[row, col]

                for s in range(top_size):
                    under_owner = stack[s]

                    if under_owner == p:
                        score += size_weights[s] * 0.3

                    elif under_owner == o:
                        score -= size_weights[s] * 0.3

        
        for p_idx in (1, 2):

            multiplier = 1 if p_idx == p else -1

            left = (
                r[p_idx, SMALL] * 20
                + r[p_idx, MEDIUM] * 40
                + r[p_idx, LARGE] * 60
            ) * 0.45

            score += left * multiplier

        return score
