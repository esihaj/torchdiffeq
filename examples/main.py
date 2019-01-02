import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
import argparse
import gym
import gym_ode
import torch.optim as optim
from odenet_mnist import create_mnist_model, get_mnist_loaders, inf_generator
from time_rnn import TimeRNN


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
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


def finish_episode(reward_list, log_prob_list):
    global optimizer
    R = 0
    policy_loss = []
    rewards = []
    for r in reward_list:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.stack(rewards)
    rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + eps)
    for log_prob, reward in zip(log_prob_list, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    return policy_loss


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

    for itr in trange(args.nepochs * batches_per_epoch):
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        env.reset(x, args.tol, args.tol)
        done = False
        # first = True
        reward_list = []
        log_prob_list = []
        done_list = []
        count = 0
        while not done and count < 10:
            #dt = env.take_action(first)
            dt, log_prob = env.take_action()
            _, reward, done, _ = env.step(dt)

            log_prob_list.append(log_prob)
            reward_list.append(reward)
            done_list.append(done)
            done = done.all().item()
            # first = False
            count += 1
        loss = finish_episode(reward_list, log_prob_list)
        # if itr % batches_per_epoch == 0:
        print("loss", loss)
