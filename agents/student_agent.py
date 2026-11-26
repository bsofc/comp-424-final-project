
"""Student agent: edit ONLY this file."""
from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES)
from constants import *
import numpy as np
import time


@register_agent("student_agent") # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self.rng = np.random.default_rng()

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play."""
        start = time.perf_counter()
        max_time = start + 1.7 # to make sure we dont pass the time 2s
        valid_moves = get_valid_moves(board, reserves, player)
        if not valid_moves:
            return random_move(board, reserves, player)

        # check if already have a winning move
        for mv in valid_moves:
            b, r = copy_state(board, reserves)
            execute_move(b, r, player, mv)
            if check_winner(b, player) == player:
                return mv

        win_length = int(WIN_LENGTH)
        lines = LINES[win_length]
        best_move = valid_moves[0] # take first one will update later anyways
        best_table = {}

        def eval_board(b, r): # FOR MINIMAX
            # state of board, reserve) evaluated from player's perspective
            score = 0.0
            center_coords = {(1, 1), (1, 2), (2, 1), (2, 2)}
            grid_size = b.shape[0]
            corner_coords = {(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)}
            size_weights = (2, 5, 10)
            threat_len = win_length - 1

            owners, sizes = top_view(b) # see top piece of all cells
            own_list = owners.tolist()
            size_list = sizes.tolist()

            #new HEURISTICS
            #line threats: add bonus when we forme winning line, substract for opponement winning line
            # https://www.cs.trincoll.edu/~ram/cpsc352/notes/heuristics.html
            for line in lines:
                player_count = 0
                opponent_count = 0
                player_large = 0
                opponent_large = 0
                blocked = False

                for row, col in line:
                    o = own_list[row][col]
                    if o == OBSTACLE: #also ignores OBSTACLE 
                        blocked = True
                        break
                    elif o == player:
                        player_count += 1
                        if size_list[row][col] == LARGE:
                            player_large += 1
                    elif o == opponent:
                        opponent_count += 1
                        if size_list[row][col] == LARGE:
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

            #give bonus if in the center (want to prioritize center) 
            for r_idx in range(grid_size):
                for c_idx in range(grid_size):
                    owner = own_list[r_idx][c_idx]
                    if owner > 0 and owner != OBSTACLE:
                        if (r_idx, c_idx) in center_coords:
                            pos_weight = 1.5
                            score += 50 if owner == player else -50
                        elif (r_idx, c_idx) in corner_coords:
                            pos_weight = 1.2
                        else:
                            pos_weight = 1.0

                        top_size = size_list[r_idx][c_idx]
                        if owner == player:
                            score += size_weights[top_size] * pos_weight
                        elif owner == opponent:
                            score -= size_weights[top_size] * pos_weight

                        stack = b[r_idx, c_idx]
                        for s in range(top_size):
                            under_owner = stack[s]
                            if under_owner == player:
                                score += size_weights[s] * 0.3
                            elif under_owner == opponent:
                                score -= size_weights[s] * 0.3

            for p_idx in (1, 2): # player 1 or 2
                if p_idx == player:
                    multiplier = 1
                else:
                    multiplier = -1
                # left in reserve, calculates for both players
                left = (r[p_idx, SMALL] * 20 + r[p_idx, MEDIUM] * 40 + r[p_idx, LARGE] * 60) * 0.45
                score += left * multiplier # -1 if min and 1 for max

            return score #we would want if player1 a very high +nb, if player2 a very low -nb

        # how we order the moves for minimax
        def score_move(owners_arr, sizes_arr, current_p, move, depth): # for curr depth on the ids
            score = 0
            other_p = opponent if current_p == player else player
            grid_size = owners_arr.shape[0]
            size_weights = (2, 5, 10)
            dest = move.dest # tuple

            if dest is not None:
                row, col = dest
                #reward gobbling an opponent piece
                if int(owners_arr[row, col]) == other_p:
                    target_size = int(sizes_arr[row, col])
                    target_weight = size_weights[target_size] if 0 <= target_size < len(size_weights) else 10
                    score += 75 + target_weight
                # heuristics
                if row in (1, 2) and col in (1, 2):
                    score += 50 # centers +50
                elif row in (0, grid_size - 1) and col in (0, grid_size - 1):
                    score += 30 # corners +30

            #new move 
            if move.src is None:
                score += 10

            if depth in best_table and move in best_table[depth]:
                score += 100
            return score

        # order the moves
        def get_ordered_moves(b, r, current_p, depth):
            moves = get_valid_moves(b, r, current_p)
            if not moves:
                return []
            owners_arr, sizes_arr = top_view(b)
            return sorted(moves, key=lambda m: score_move(owners_arr, sizes_arr, current_p, m, depth), reverse=True)
        #used sorted with lambda cite:
        # https://www.varsitytutors.com/learner/subjects/python/lessons/sorted-with-key-functions

        # used the geeks pseudocode, also saw the code (also AI lol to make it nicer and safer?)
        # https://www.geeksforgeeks.org/dsa/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
        def minimax(b, r, depth, alpha, beta, maximizing_player, current_player, last_mover):
            if time.perf_counter() >= max_time: # safer to make sure dont exceed time (risk a random move otw)
                raise TimeoutError("Search time exceeded") #suggesteed by AI for safety

            w = check_winner(b, last_mover) # check if already have a winner without rec
            if w == player:
                return 10000 + depth, None
            if w == opponent:
                return -10000 - depth, None
            # want a high depth bc means we look more ahead
            if depth == 0: # no winner yet, look at board
                return eval_board(b, r), None
            # get valid moves and order them
            valid_moves_node = get_ordered_moves(b, r, current_player, depth)
            if not valid_moves_node:
                return 0, None
            # choose first one, wil update later
            best_m = valid_moves_node[0]
            # we are max rn
            if maximizing_player:
                max_eval = -float('inf')
                for move in valid_moves_node:
                    nb, nr = copy_state(b, r) # get a cop to try it out
                    execute_move(nb, nr, current_player, move)

                    # min's turn sim
                    eval_score, _ = minimax(nb, nr, depth - 1, alpha, beta, False, opponent if current_player == player else player, current_player)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_m = move
                    alpha = max(alpha, eval_score) # update alpha
                    # prunning part
                    if beta <= alpha:
                        if depth not in best_table:
                            best_table[depth] = [] # record on best_table (no other good moves at this depth)
                        if move not in best_table[depth]:
                            best_table[depth].insert(0, move) # insert currmove to best_table at beginning
                        if len(best_table[depth]) > 2:
                            best_table[depth].pop()
                        break
                return max_eval, best_m
            # we are min , similar to max part
            else:
                min_eval = float('inf')
                for move in valid_moves_node:
                    nb, nr = copy_state(b, r)
                    execute_move(nb, nr, current_player, move)

                    eval_score, _ = minimax(nb, nr, depth - 1, alpha, beta, True, opponent if current_player == player else player, current_player)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_m = move
                    beta = min(beta, eval_score)
                    # prunning part
                    if beta <= alpha:
                        if depth not in best_table:
                            best_table[depth] = []
                        if move not in best_table[depth]:
                            best_table[depth].insert(0, move)
                        if len(best_table[depth]) > 2:
                            best_table[depth].pop()
                        break
                return min_eval, best_m

        # IDS for search helps with timing :)
        max_depth = 4 # could change but within time, this has worked so far
        for current_depth in range(1, max_depth + 1): # look ahead up to max
            try:
                _, move = minimax(board, reserves, current_depth, -float('inf'), float('inf'), True, player, opponent)
                if move is not None and move in valid_moves:
                    best_move = move
            except TimeoutError: # if run out of time, break
                break
            if time.perf_counter() >= max_time: # second break in case super close of 2s
                break
        return best_move