"""
This module contains agents that play reversi.

Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2



_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color
    
    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also available at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(
                target=self.search, 
                args=(
                    self._color, board, valid_actions, 
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
                self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')

"""Reversi AI Completition: Implement an agent that plays Reversi and beat your friends
- An evaluation function
- Depth-limited Minimax with Alpha-Beta pruning

Competition Rules
- Each turn, your AI has only 10 seconds to provide a move, otherwise, a random move will be made.
- Your AI must not search beyond 10 steps.
- Your evaluation function should not perform searching (e.g. Monte Carlo simulation).

Members of the winner team gets a drink at Starbucks (when things are back to normal)"""

class RandomAgent(ReversiAgent):
    """An agent that move randomly."""
    
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

class BakaAgent(ReversiAgent): 
    """An agent made by player."""
    
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])
        # transition(board, self.player #สำหรับตาแรก #ควรจะสลับเพลเยอร์เองด้วย, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        # while True:
        # pass

        # print only action, as function return both eval value and action, do v, action to prevent TypeError: only size-1 arrays can be converted to Python scalars.
        v, best_action = self.minimax(board, valid_actions, 0, float('-inf'), float('inf'), True) 

        if best_action is not None:
            output_move_row.value = best_action[0]
            output_move_column.value = best_action[1]

    def minimax(self, board, valid_actions, depth, alpha, beta, maximizingPlayer):
        # Pseudo Code Reference: https://www.javatpoint.com/ai-alpha-beta-pruning

        #set depth limit to 4
        if depth == 4: # if depth reaches limit/terminal node, return evaluation-utility 
            return self.eval_function(board)

        best_action = None
        if maximizingPlayer:
            max_v = float('-inf')

            for action in valid_actions:
                new_board = transition(board, self.player, action)
                new_action = self.possible_moves(new_board, self.switch_player(self.player))
                v = self.minimax(new_board, new_action, depth + 1, alpha, beta, False)

                if v > max_v: #find max-value
                    max_v = v
                    best_action = action

                alpha = max(alpha, max_v)

                if beta <= alpha: #prunned
                    break
            
            #print(max_v, best_action)
            
            if depth != 0: #if-else for prevent case'>' not supported between instances of 'tuple' and 'float'
                return max_v
            else:
                return max_v, best_action #evalpont + best action position x,y 

        else:
            min_v = float('inf')

            for action in valid_actions:
                new_board = transition(board, self.switch_player(self.player), action)
                new_action = self.possible_moves(new_board, self.player)
                v = self.minimax(new_board, new_action, depth + 1, alpha, beta, True)

                if v < min_v: #find min-value
                    min_v = v 
                    best_action = action

                beta = min(beta, min_v)

                if beta <= alpha: #prunned
                    break
            
            #print(min_v, best_action)

            if depth != 0: #if-else for prevent case'>' not supported between instances of 'tuple' and 'float'
                return min_v
            else:
                return min_v, best_action

    def switch_player(self, player):
        if player == 1: #black to white
            return player * -1 # multiply -1 EZ 
        else: #white to black
            return player * -1
        
    def possible_moves(self, board, player):

        valids = _ENV.get_valid((board, player)) # from env.py library - Get all valid locations for the current state.
        valids = np.array(list(zip(*valids.nonzero()))) # find all the non-zero positions for valid moves in the board.
        return valids
    
    def eval_function(self, board):
        # Evaluation Reference: http://dhconnelly.com/paip-python/docs/paip/othello.html
        
        # calculates the evaluation score after reaching depth limit
        # using weighing table to evaluate score. # corners are more likely to get advantage in reversi.
        weight_board = [
            100, -20,  10,   5,   5,  10, -20, 100,   
            -20, -50,  -2,  -2,  -2,  -2, -50, -20,   
             10,  -2,  -1,  -1,  -1,  -1,  -2,  10,   
              5,  -2,  -1,  -1,  -1,  -1,  -2,   5,   
              5,  -2,  -1,  -1,  -1,  -1,  -2,   5,   
             10,  -2,  -1,  -1,  -1,  -1,  -2,  10,   
            -20, -50,  -5,  -5,  -5,  -5, -50, -20,   
            100, -20,  20,   5,   5,  20, -20, 100
        ]

        self.weights = np.array(weight_board).reshape(8,8) # shape the array to numpy first 
        evaluate_board = np.array(list(zip(*board.nonzero()))) # use this to find all the non-zero positions in the board to evaluate score. #also zip the positions together in list.
        
        agent_score = 0
        opponent_score = 0
        for position in evaluate_board: 
            x, y = position[0], position[1]
            if board[x][y] == self._color:
                agent_score += self.weights[x][y]
            else:
                opponent_score += self.weights[x][y]

        # evaluate score is difference between sum weights of player and the opponent player.
        score = agent_score - opponent_score 
        return score  

        
