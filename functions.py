import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def running_mean(x,N=200):
    """returns running mean of list x with window size N"""
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

def plot_losses(ax, losses, tot_rewards, game_lengths, m_outcomes, epoch, smoothing_window=50):
    """plots losses, rewards, length of game and fraction of games won during training"""
    losses = np.array(losses)
    tot_rewards = np.array(tot_rewards)
    game_lengths = np.array(game_lengths)
    [ax[i,j].cla() for i in range(2) for j in range(2)]
    ax[0,0].semilogy(running_mean(losses, min([epoch,smoothing_window])))
    ax[0,0].set_ylim([1e-1,1e+1])
    ax[0,0].set_xlabel("Epochs",fontsize=13)
    ax[0,0].set_ylabel("Loss",fontsize=13)
    ax[0,1].plot(running_mean(tot_rewards, min([epoch,smoothing_window])))
    ax[0,1].set_ylim(-2,6)
    ax[0,1].set_ylabel("Reward", fontsize=13)
    ax[1,0].plot(running_mean(game_lengths, min([epoch,smoothing_window])))
    ax[1,0].set_ylim(0, 1)
    ax[1,0].set_ylabel("Proportion of games lost", fontsize=13)
    ax[1,1].plot(m_outcomes)
    ax[1,1].set_ylim(0, 1)
    ax[1,1].set_ylabel("Proportion of games won", fontsize=13)
    plt.show(block=False)
    plt.pause(0.001)

def policy(qval, epsilon=0):
    """chooses action given q-values using epsilon-greedy policy"""
    if random.random() < epsilon:
        action = np.random.randint(low=0, high=9)
    else:
        action = np.argmax(qval)
    return action

def train(model, target_net, replay, loss_fn, optimizer, env, ax=None, logger=None, **params):
    """training loop for DQN with experience replay and target network. Plotting and logging are optional."""
    sync_idx = 0
    for i in range(params['epochs']):
        env.reset()
        state1_ = env.state.reshape(1, 18)
        state1 = torch.from_numpy(state1_).float()
        done = False
        tot_reward = 0
        game_round = 0
        if env.current_player == 1:
            state2_, reward2, done = env.make_move(env.random_move(), render=False)
        while not done:
            sync_idx += 1
            game_round += 1
            qval = model(state1)
            qval_ = qval.squeeze().data.numpy()
            if random.random() < params['epsilon']:
                action = np.random.randint(low=0, high=9)
            else:
                action = np.argmax(qval_)
            state2_, reward, done = env.make_move(action, render=False)
            if not done:
                state2_, reward, done = env.make_move(env.random_move(), render=False)
            state2_ = state2_.reshape(1, 18)
            state2 = torch.from_numpy(state2_).float()
            tot_reward += reward
            exp = (state1, action, reward, state2, done)
            replay.add_to_memory(exp)
            state1 = state2

            if len(replay) > params['batch_size']:
                state1_batch, action_batch, reward_batch, state2_batch, done_batch = replay.sample_batch()
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = target_net(state2_batch)
                maxQ = torch.max(Q2, dim=1)[0]
                Y = reward_batch + (1 - done_batch) * params['gamma'] * maxQ
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if sync_idx % params['sync_freq'] == 0:
                    target_net.load_state_dict(model.state_dict())

        if len(replay)>params['batch_size'] and logger!=None:
            logger.log(loss.item(), tot_reward, game_round, reward)

        if ax.any()!=None and logger!=None:
            if i % 1000 == 0 and i >= params['batch_size']:
                plot_losses(ax, logger.losses, logger.tot_rewards, logger.m_lost_games, logger.m_outcomes, i,
                            smoothing_window=100)

        if i % 1000 == 0:
            print('epoch: ', i)
            print('eps: ', round(params['epsilon'], 4))
            print('reward: ', reward)
            print('Q: ', np.max(abs(qval_)))
            print('sync_idx: ', sync_idx)
            print('-' * 50)