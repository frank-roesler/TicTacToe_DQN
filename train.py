from classes import *
from environment import TicTacToe
from functions import *
import copy

l1 = 18
l2 = 256
l3 = 256
l4 = 9

model = ShallowNet(l1,l2,l3,l4).float()
model2 = copy.deepcopy(model)
model2.load_state_dict(model.state_dict())

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

ttt = TicTacToe(mode='PvM')

params = {
    'epsilon': 0.01,
    'gamma': 0.99,
    'epochs': 6501,
    'batch_size': 500,
    'sync_freq': 500,
}

replay = ExperienceReplay(mem_size=1000, batch_size=params['batch_size'])
logger = Logger()
fig, ax = plt.subplots(2,2, constrained_layout=True, figsize = (10,8))
model.train()

train(model, model2, replay, loss_fn, optimizer, ax=ax, logger=logger, env=ttt, **params)

print(np.mean(logger.game_outcomes[-1000:-1]))
torch.save(model.state_dict(), 'state_dict_trained')
plot_losses(ax,logger.losses,logger.tot_rewards,logger.m_lost_games,logger.m_outcomes,params['epochs'],smoothing_window=10)
plt.show()







