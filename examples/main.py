import torch
import torch.nn as nn
from tqdm import trange
import argparse
import gym
import gym_ode
from odenet_mnist import create_mnist_model, get_mnist_loaders, inf_generator
from time_rnn import TimeRNN

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--iter_save_period', type=int, default=100000)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--network_size', type=int, default=16)

parser.add_argument('--model_path', type=str, default='model.pth')

args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, feature_layers = create_mnist_model()
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    env = gym.make('ode-v0')
    env.init(device, model, TimeRNN(7 * 7 * 16 * 2 + 1))

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
        print('new batch')
        # first = True
        reward_list = []
        done_list = []
        while not done:
            #dt = env.take_action(first)
            dt = env.take_action()
            _, reward, done, _ = env.step(dt)
            reward_list.append(reward)
            done_list.append(done)
            done = done.all().item()
            # first = False
