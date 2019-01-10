import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
import argparse
import gym
import gym_ode
import torch.optim as optim
import pickle
from odenet_mnist import create_mnist_model, get_mnist_loaders, inf_generator
from time_rnn import TimeRNN

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--iter_save_period', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--network_size', type=int, default=16)

parser.add_argument('--model_path', type=str, default='model.pth')

args = parser.parse_args()

eps = np.finfo(np.float32).eps.item()


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.zeros(labels.size(0), C).to(labels)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def finish_episode(reward_list, done_list, log_prob_list):
    global optimizer
    episode_length = len(done_list)
    done_list = torch.stack(done_list, dim=0)
    first_done = done_list.argmin(dim=0) + 1
    first_done[first_done == episode_length] = 0
    first_done = make_one_hot(first_done.unsqueeze(1), episode_length)
    done_list = torch.where(first_done.transpose(1, 0) == 1, torch.zeros_like(done_list), done_list)
    done_list = done_list.view(-1)
    R = 0
    rewards = []
    for r in reward_list:
        R = r + args.gamma * R
        rewards.append(R)  # rewards.insert(0, R)
    rewards = torch.cat(rewards, dim=0)
    rewards = rewards[1 - done_list]
    rewards_mean = rewards.mean().item()
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    log_prob = torch.cat(log_prob_list, dim=0)
    log_prob = log_prob[1 - done_list]
    optimizer.zero_grad()
    policy_loss = (-log_prob * rewards).sum()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item(), rewards_mean


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, feature_layers = create_mnist_model()
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    env = gym.make('ode-v0')
    env.init(device, model, TimeRNN(7 * 7 * 16 * 2 + 1))

    optimizer = optim.Adam(env.dt_rnn.parameters(), lr=args.lr)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        False, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    all_losses = []
    all_reward_means = []

    for itr in trange(args.nepochs * batches_per_epoch):
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        env.reset(x, args.tol, args.tol)
        done = False
        reward_list = []
        log_prob_list = []
        done_list = []
        count = 0
        while not done and count < 10:
            dt, log_prob = env.take_action()
            prev_done = done_list[-1] if count > 0 else None
            _, reward, done, _ = env.step(dt, prev_done)

            log_prob_list.append(log_prob)
            reward_list.append(reward)
            done_list.append(done)
            done = done.all().item()
            count += 1
        loss, reward_mean = finish_episode(reward_list, done_list, log_prob_list)
        all_losses.append(loss)
        all_reward_means.append(reward_mean)
        if itr % 100 == 0:
            with open("loss_and_mean.pkl", "rb") as f:
                pickle.dump({"loss": all_losses, "mean": all_reward_means}, f)
                print("saved[%d]" % itr)
