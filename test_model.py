from environment import TicTacToe
from classes import *

l1 = 18
l2 = 256
l3 = 256
l4 = 9

model = ShallowNet(l1,l2,l3,l4).float()
state_dict = torch.load('state_dict_trained')
model.load_state_dict(state_dict)

ttt = TicTacToe(mode='PvM')
ttt.start_game(model=model)
