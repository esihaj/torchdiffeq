import gym
import torch
import torch.nn as nn
from torchdiffeq._impl.dopri5 import _DORMAND_PRINCE_SHAMPINE_TABLEAU
from torchdiffeq._impl.rk_common import _runge_kutta_step
from torchdiffeq._impl.misc import _compute_error_ratio, _is_iterable, _select_initial_step, _optimal_step_size
from gym import error, spaces, utils
from gym.utils import seeding

INPUT_SIZE = 7*7*16*2+1

MODEL_LAYERS_NUM_ODE_BLOCK = 3
MODEL_LAYERS_NUM_DOWN_SAMPLE_START = 0
MODEL_LAYERS_NUM_DOWN_SAMPLE_END = 2
ODE_BLOCK_LAYER_NUM_FUNC = 0

REWARD_ACCEPT = -1.0
REWARD_REJECT = -1000.0
DONE_TIME = 1.0


class OdeEnv(gym.Env):
    metadata = {'render.modes': None}
    distribution = torch.distributions.normal.Normal
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
    dt_rnn = None
    init_net = None
    func_net = None

    def init(self, device, base_model, dt_rnn):
        self.dt_rnn = dt_rnn
        base_layers = list(base_model.children())
        self.init_net = nn.Sequential(
            *base_layers[MODEL_LAYERS_NUM_DOWN_SAMPLE_START : MODEL_LAYERS_NUM_DOWN_SAMPLE_END+1]
            ).to(device)
        func_net = list(base_layers[MODEL_LAYERS_NUM_ODE_BLOCK].children())[ODE_BLOCK_LAYER_NUM_FUNC]
        self.func_net = lambda t, y: (func_net(t, y[0]),)

    def reset(self, x, atol, rtol):
        self.dt_rnn.reset()

        t0 = torch.tensor([0.0]).repeat(x.size(0))#.double()
        t0 = t0.view(-1, 1, 1, 1)
        y0 = self.init_net(x)
        f0 = self.func_net(t0[0].type_as(y0[0]), (y0, ))
        self.observation_space = [[t0, (y0, ), f0], [[t0, y0, f0[0]]]]

        # TODO not really the right place for these, but where should I put them?
        self.rtol = self.rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = self.atol if _is_iterable(atol) else [atol] * len(y0)

    def take_action(self, is_first=False):
        # if is_first:
        #     t0, y0, f0 = self.observation_space[0]
        #     return _select_initial_step(self.func_net, t0[0], y0, 4, self.rtol[0], self.atol[0], f0=f0).to(t0).repeat(y0[0].size(0)).double()
        # else:
        #     return _optimal_step_size(self.last_action[0].double(), self.mean_sq_error_ratio).repeat(self.last_action.size(0)).double()
        mu, sigma = self.dt_rnn(self.observation_space[1])
        dist = self.distribution(mu, sigma)
        dt = dist.sample()
        return dt, dist.log_prob(dt)

    def step(self, action, prev_done=None):
        t0 = self.observation_space[0][0]
        y0 = self.observation_space[0][1]
        f0 = self.observation_space[0][2]

        # self.last_action = action
        action_reshaped = action.view(-1, 1, 1, 1)
        # print("call rk", y0.size())
        y1, f1, y1_error, k, observations = _runge_kutta_step(self.func_net, y0, f0, t0, action_reshaped, self.tableau)
        t1 = t0+action_reshaped
        self.observation_space = ((t1, y1, f1), observations)

        reward, done = self.calculate_reward(y1_error, y0, y1)
        done = (((t1 > DONE_TIME).squeeze() + (1 - done)) > 0)
        if prev_done is not None:
            new_done = (done + prev_done) > 0
        else: new_done = done

        return observations, reward, new_done, None

    def calculate_reward(self, y1_error, y0, y1):
        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = mean_sq_error_ratio[0] <= 1
        # self.mean_sq_error_ratio = mean_sq_error_ratio
        accept_step_float = accept_step.float()
        return accept_step_float * REWARD_ACCEPT + (1.0 - accept_step_float) * REWARD_REJECT, accept_step

    def render(self, mode='human'):
        pass
