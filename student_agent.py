"""Student agent: edit ONLY this file."""
from tracemalloc import start

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view)
from constants import *
import numpy as np
import time

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

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

        # create min-max tree for now
        # generate all possible boards state until a win
        depth = 0 # depth for the minimax tree
        alpha = -(float('inf')) # alpha for alpha-beta pruning
        beta = (float('inf')) # beta for alpha-beta pruning
        STEP_TIME_LIMIT = 2.0  # seconds                   
        margin = 0.1  # seconds
        maximum_depth = 1 #random for now

        # use iterative deepening to search for the best move within the time limit
        
        # make a set to keep track of boards that have already been evaluated to avoid re-evaluating them
        

        def no_time_left():
            if (time.perf_counter() - start) > (STEP_TIME_LIMIT - margin):
                return True
       
        def minimax(b, r, p, o, depth, alpha, beta, maximum_depth):

            # check if there is a winner? check with the state of opponent since it was just the opponent's turn
            winner = check_winner(b, o)
            if winner == p:
                return (10000 - 3*depth)
            elif winner == o:
                return -(10000 - 3*depth)

            # check if the depth reached the limit
            if depth >= maximum_depth: # random depth limit
                # use a heuristic to evaluate the board state and give score
                return evaluate(b, p, o)

            # want the earliest win if possible, so if the score is the same, choose the one with the least depth
            # so make a score with depth in mind

            best = -(float('inf')) 
            score = -(float('inf')) 
            # look at the board first and get all possible moves
            moves = get_valid_moves(b, r, p) 
            if len(moves) == 0:
                # no possible moves? so it loses?
                return -(10000 - 3*depth)

            # generate all possible board states with those moves
            for m in moves:
                if no_time_left():
                    raise TimeoutError

                
                # make copy of board
                bo, re = copy_state(b, r)
                execute_move(bo, re, p, m)

                # check if it's a win
                winner = check_winner(bo, p)
                if winner ==  p:
                    # we won so this move is good so use it
                    score = 10000 - 3*depth
                elif winner == o:
                    # this move will make the opponent win, so don't use in this round
                    score = -(10000 - 3*depth)
                # else if the board is full but no winner then +0
                # else, go in depth
                else:
                    score = -minimax(bo, re, o, p, depth+1, -beta, -alpha, maximum_depth) # minus since it'll be in the pov of the opponent and swap alpha and beta for pov opponent

               
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break # pruning

            return best
        

        diagonals = [[(0,0), (1,1), (2,2), (3,3)], [(0,3), (1,2), (2,1), (3,0)]]
        horizontals = [[(0,0), (0,1), (0,2), (0,3)], [(1,0), (1,1), (1,2), (1,3)], [(2,0), (2,1), (2,2), (2,3)], [(3,0), (3,1), (3,2), (3,3)]]
        verticals = [[(0,0), (1,0), (2,0), (3,0)], [(0,1), (1,1), (2,1), (3,1)], [(0,2), (1,2), (2,2), (3,2)], [(0,3), (1,3), (2,3), (3,3)]]
        
        winning_lines = diagonals + horizontals + verticals
        # if you have pieces in the middle cells, give 3 points since it can create three different type of winning lines
        middle_cells = [(1,1), (2,2), (1,2), (2,1)]
        middle_cells_score = 3
        # same for corner cells
        corner_cells = [(0,0), (0,3), (3,0), (3,3)]
        corner_cells_score = 3
        # outer_corner cells will give 2 points since it can create two different type of winning lines
        outer_corner_cells = [(0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2)]
        outer_corner_cells_score = 2

        # different sizes will have different points
        big_piece_score = 6
        medium_piece_score = 3
        small_piece_score = 1

        def evaluate(b, p, o) -> int:
            # evaluate the board state
            # use heuristics

            player_score = 0
            opponent_score = 0

            players, sizes = top_view(b)

            # also add heurisitic for a closer win    

            # go through each cell and check if it's the player or opponent and add score 
            for i in range(4):
                for j in range(4):
                    # check each cell
                    if(players[i][j] == p):
                        # check size of the piece and add score
                        size = sizes[i][j]
                        if(size == SMALL):
                            player_score += small_piece_score
                        elif(size == MEDIUM):
                            player_score += medium_piece_score
                        elif(size == LARGE):
                            player_score += big_piece_score
                        # check if it's in the middle, corner, or outer corner and add score
                        if((i,j) in corner_cells):
                            player_score += corner_cells_score
                        elif((i,j) in outer_corner_cells):
                            player_score += outer_corner_cells_score
                        elif((i,j) in middle_cells):
                            player_score += middle_cells_score
                    elif(players[i][j] == o):
                        # check size of the piece and add score
                        size = sizes[i][j]
                        if(size == SMALL):
                            opponent_score += small_piece_score
                        elif(size == MEDIUM):
                            opponent_score += medium_piece_score
                        elif(size == LARGE):
                            opponent_score += big_piece_score
                        # check if it's in the middle, corner, or outer corner and add score
                        if((i,j) in corner_cells):
                            opponent_score += corner_cells_score
                        elif((i,j) in outer_corner_cells):
                            opponent_score += outer_corner_cells_score
                        elif((i,j) in middle_cells):
                            opponent_score += middle_cells_score

            # check for winning lines and add score if there are 2 or 3 in a row for the player or opponent
            for line in winning_lines:
                # check each line who is closer to winning with that line
                player_count = 0
                opponent_count = 0
                for cell in line:
                    if players[cell[0]][cell[1]] == p:
                        player_count += 1
                    elif players[cell[0]][cell[1]] == o:
                        opponent_count += 1
                if player_count == 3:
                    player_score += 100
                elif player_count == 2:
                    player_score += 50
                elif player_count == 1:
                    player_score += 10

                if opponent_count == 1:
                    opponent_score += 10
                elif opponent_count == 2:
                    opponent_score += 50
                elif opponent_count == 3:
                    opponent_score += 100

            return player_score - opponent_score # negative mean opponent is winning

        

        def process(board, reserves, player, opponent, alpha, beta, depth, maximum_depth):
            
            scores = []
            best_score = -(float('inf'))
            # look at the board first and get all possible moves
            moves = get_valid_moves(board, reserves, player) # not sure I understand the boolean thing
            
            if len(moves) == 0:
                return random_move(board, reserves, player) # no possible moves? so it loses?
            else: 
                best_m = moves[0] # random move for now

            best_move = moves[0]

       
            for i in range(1, 10):
                ms = moves.copy()
                if best_move in ms:
                    ms.remove(best_move)
                    ms.insert(0, best_move) # put the best move at the front of the list so it gets evaluated first
                

                maximum_depth = i
                alpha = -(float('inf')) # reset for each ids round
                beta = (float('inf'))    
                scores = [] # reset for each ids round
                # generate all possible board states with those moves


                best_m = None

                try:
                    for m in ms:
                        if no_time_left():
                            raise TimeoutError
                        # make copy of board
                        b, r = copy_state(board, reserves)
                        execute_move(b, r, player, m)
                        # check if it's a win
                        winner = check_winner(b, player)
                        if winner == player:
                            # we won so this move is good so use it
                            return m
                        elif winner == opponent:
                            # this move will make the opponent win, so don't use in this round
                            continue
                        else:
                            # nobody wins this round, so we can go in depth to see what will happen
                            if beta <= alpha:
                                break # pruning
                            score = -minimax(b, r, opponent, player, depth+1, -beta, -alpha, maximum_depth) # negative since next turn is opponent's turn and we want to see if it's good for us or not
                            scores.append((score, m))
                            alpha = max(alpha, score)
                            if beta <= alpha:
                                break # pruning

                    
                # choose the move with the highest score
                    best_score = -(float('inf'))
                    for s in scores:
                        if best_score < s[0]:
                            best_m = s[1] # update best move
                            best_score = s[0] # update best score
                    if best_m is not None:
                        best_move = best_m # update best move for this iteration depth
                        complete_depth = i # update the completed depth
                
                except TimeoutError:
                    break # break out of the loop if time is up

              
            return best_move
        return process(board, reserves, player, opponent, alpha, beta, depth, maximum_depth)
       
