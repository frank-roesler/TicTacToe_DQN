import numpy as np
import torch
import random

class TicTacToe:
    """Implementation of classic TicTacToe game"""
    def __init__(self, mode='PvM'):
        self.current_player = random.randint(0,1) # 0='X' (computer), 1='O' (human)
        self.state = np.zeros((3,3,2))
        self.mode = mode
        self.done = False

    def make_move(self, move, render = True):
        if move in self.possible_moves():
            y = move % 3
            x = int((move - y) / 3)
            self.state[x,y,self.current_player] = 1
            if render:
                self.render()
            if self.current_player==0:
                reward = 1
            else:
                reward = 0
            reward += self.check_victory(render)
            self.current_player = (self.current_player+1)%2
        else:
            if render:
                print('Invalid move by player',self.current_player,'. Game is over.')
            if self.current_player == 0: # Only penalize bad behaviour by actor; not by opponent.
                reward = -1
            else:
                reward = 0
            self.done = True
            self.state = np.zeros((3,3,2)) # Return auxiliary "next state" for training
        return self.state, reward, self.done

    def possible_moves(self):
        available_positions = (self.state[:, :, 0] == 0) * (self.state[:, :, 1] == 0)
        available_indices = np.where(available_positions.reshape(-1))[0]
        return available_indices

    def random_move(self):
        available_indices = self.possible_moves()
        move = np.random.choice(available_indices)
        return move

    def render(self):
        board = np.zeros((3,3), dtype='str')
        for i in range(3):
            for j in range(3):
                board[i,j] = ' '
        board[self.state[:,:,0]!=0] = 'X'
        board[self.state[:,:,1]!=0] = 'O'
        print('-' * 50)
        print('Player ', self.current_player)
        print(board)
        print('-'*50)

    def check_victory(self,render):
        arr0 = self.state[:,:,self.current_player]
        p_row0 = np.prod(arr0, axis=0).any()
        p_col0 = np.prod(arr0, axis=1).any()
        p_diag0 = any([np.prod(np.diag(arr0)), np.prod(np.diag(np.fliplr(arr0)))])
        arr1 = self.state[:, :, (self.current_player+1)%2]
        p_row1 = np.prod(arr1, axis=0).any()
        p_col1 = np.prod(arr1, axis=1).any()
        p_diag1 = any([np.prod(np.diag(arr1)), np.prod(np.diag(np.fliplr(arr1)))])
        if any([p_row0,p_col0,p_diag0,p_row1,p_col1,p_diag1])==1:
            if render:
                print('GAME OVER!')
                if self.current_player==0:
                    print('Robot wins!')
                else:
                    print('Human wins!')
            self.done = True
            if self.current_player == 0:
                return 5
            else:
                return -5
        elif len(self.possible_moves())==0:
            if render:
                print('GAME OVER!')
                print('Draw!')
            self.done = True
            return 0
        else:
            return 0

    def start_game(self, model=None):
        if self.mode == 'PvM':
            if model==None:
                print('You are Player 1.')
                if self.current_player==1:
                    print('Human begins.')
                    while not self.done:
                        move = int(input('Enter move (0-8): '))
                        self.make_move(move) # Player move
                        if self.done: break
                        move = self.random_move()
                        self.make_move(move) # Computer move
                else:
                    print('Computer begins.')
                    while not self.done:
                        move = self.random_move()
                        self.make_move(move)  # Computer move
                        if self.done: break
                        move = int(input('Enter move (0-8): '))
                        self.make_move(move)  # Player move
            else:
                print('You are Player 1.')
                if self.current_player==1:
                    print('Human begins.')
                    while not self.done:
                        # Player move:
                        move = int(input('Enter move (0-8): '))
                        self.make_move(move)
                        if self.done: break
                        # Computer move:
                        state_ = self.state.reshape(1, 18)
                        state = torch.from_numpy(state_).float()
                        qval = model(state).squeeze().data.numpy()
                        ctr=0
                        move = np.argmax(qval)
                        while not move in self.possible_moves(): # If model chooses forbidden move, make it choose again
                            qval[move] = min(qval) - 1
                            move = np.argmax(qval)
                            print('attempt: ', ctr)
                            ctr+=1
                        self.make_move(move)
                else:
                    print('Computer begins.')
                    while not self.done:
                        # Computer move:
                        state_ = self.state.reshape(1, 18)
                        state = torch.from_numpy(state_).float()
                        qval = model(state).squeeze().data.numpy()
                        ctr = 0
                        move = np.argmax(qval)
                        while not move in self.possible_moves():  # If model chooses forbidden move, make it choose again
                            qval[move] = min(qval) - 1
                            move = np.argmax(qval)
                            print('attempt: ', ctr)
                            ctr += 1
                        self.make_move(move)
                        if self.done: break
                        # Player move:
                        move = int(input('Enter move (0-8): '))
                        self.make_move(move)
        elif self.mode == 'PvP':
            while not self.done:
                print('Player ',self.current_player)
                move = int(input('Enter move (0-8): '))
                self.make_move(move)
        else:
            print('Not a valid mode of play!')

    def reset(self):
        self.done = False
        self.current_player = random.randint(0,1)
        self.state = np.zeros((3, 3, 2))
















