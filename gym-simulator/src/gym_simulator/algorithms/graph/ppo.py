from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from gym_simulator.algorithms.graph.job_actor_ import JobActor
from gym_simulator.algorithms.graph.machine_actor import MachineActor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def permute_rows(x):
    """
    x is a np array
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


class FJSPDataset(torch.utils.data.Dataset):
    def __init__(self, n_j, n_m, low, high, num_samples=1000000, seed=None, offset=0, distribution=None):
        super(FJSPDataset, self).__init__()

        self.data_set = []
        if seed != None:
            np.random.seed(seed)
        time0 = np.random.uniform(low=low, high=high, size=(num_samples, n_j, n_m, n_m - 1))
        time1 = np.random.uniform(low=0, high=high, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = permute_rows(times[j][i])
            # Sample points randomly in [0, 1] square
        self.data = np.array(times)
        self.size = len(self.data)

    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.adj_mb, self.fea_mb, self.candidate_mb = [], [], []
        self.mask_mb, self.a_mb, self.r_mb, self.done_mb = [], [], [], []
        self.job_logprobs, self.mch_logprobs, self.mask_mch = [], [], []
        self.first_task, self.pre_task, self.action, self.mch = [], [], [], []
        self.dur, self.mch_time = [], []


def init_weights(net, scheme="orthogonal"):
    for param in net.parameters():
        if len(param.size()) >= 2:
            if scheme == "orthogonal":
                nn.init.orthogonal_(param)
            elif scheme == "normal":
                nn.init.normal_(param, std=1e-2)
            elif scheme == "xavier":
                nn.init.xavier_normal_(param)


def adv_normalize(adv):
    std = adv.std()
    assert std != 0.0 and not torch.isnan(std), "Need nonzero std"
    return (adv - adv.mean()) / (std + 1e-8)


def g_pool_cal(batch_size, n_nodes, device):
    # Set the fill value based on graph_pool_type
    fill_value = 1 / n_nodes
    elem = torch.full((batch_size * n_nodes,), fill_value, dtype=torch.float32, device=device)

    # Generate indices for sparse matrix
    idx_0 = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
    idx_1 = torch.arange(n_nodes * batch_size, device=device)
    idx = torch.stack((idx_0, idx_1))

    # Create the sparse graph pool matrix
    graph_pool = torch.sparse_coo_tensor(idx, elem, (batch_size, n_nodes * batch_size), device=device)
    return graph_pool


class PPO:
    def __init__(
        self,
        lr,
        gamma,
        k_epochs,
        eps_clip,
        decay_step_size,
        decay_ratio,
        n_j,
        n_m,
        num_layers,
        input_dim,
        hidden_dim,
        num_mlp_layers_feature_extract,
        num_mlp_layers_critic,
        hidden_dim_critic,
        vloss_coef,
        ploss_coef,
        entloss_coef,
        device,
        **kwargs
    ):

        self.n_j = n_j
        self.n_m = n_m
        self.lr = lr
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.vloss_coef = vloss_coef
        self.ploss_coef = ploss_coef
        self.entloss_coef = entloss_coef

        # Define policy networks and optimizers
        self.policy_job = JobActor(
            n_jobs=n_j,
            n_machines=n_m,
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_extract_num_mlp_layers=num_mlp_layers_feature_extract,
            critic_num_mlp_layers=num_mlp_layers_critic,
            critic_hidden_dim=hidden_dim_critic,
            device=device,
        )
        self.policy_mch = MachineActor(
            n_jobs=n_j,
            n_machines=n_m,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)

        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())

        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)

        self.job_scheduler = torch.optim.lr_scheduler.StepLR(
            self.job_optimizer, step_size=decay_step_size, gamma=decay_ratio
        )
        self.mch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.mch_optimizer, step_size=decay_step_size, gamma=decay_ratio
        )

        self.MSE = nn.MSELoss()

    def update(self, memories, batch_size, decay_flag):
        rewards_all_env = []

        # Calculate rewards for each environment
        for i in range(batch_size):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(
                reversed(memories.r_mb[0][i].tolist()), reversed(memories.done_mb[0][i].tolist())
            ):
                discounted_reward = 0 if is_terminal else reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)
        g_pool_step = g_pool_cal(
            batch_size=torch.Size([batch_size, self.n_j * self.n_m, self.n_j * self.n_m]),
            n_nodes=self.n_j * self.n_m,
            device=device,
        )

        for _ in range(self.k_epochs):
            job_log_prob, mch_log_prob, val, job_entropy, mch_entropies = [], [], [], [], []
            job_log_old_prob, mch_log_old_prob = memories.job_logprobs[0], memories.mch_logprobs[0]
            env_mask_mch, env_dur = memories.mask_mch[0], memories.dur[0]
            pool = None

            for i, (env_fea, env_adj, env_candidate, env_mask, a_index, env_mch_time, old_action, old_mch) in enumerate(
                zip(
                    memories.fea_mb,
                    memories.adj_mb,
                    memories.candidate_mb,
                    memories.mask_mb,
                    memories.a_mb,
                    memories.mch_time,
                    memories.action,
                    memories.mch,
                )
            ):
                a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(
                    x=env_fea,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=env_adj,
                    candidate=env_candidate,
                    mask=env_mask,
                    mask_mch=env_mask_mch,
                    dur=env_dur,
                    a_index=a_index,
                    old_action=old_action,
                    mch_pool=pool,
                    old_policy=False,
                )

                pi_mch, pool = self.policy_mch(action_node, hx, mask_mch_action, env_mch_time)
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)
                val.append(v)
                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)

            job_loss_sum, mch_loss_sum = self.compute_loss(
                batch_size, job_log_prob, mch_log_prob, val, rewards_all_env, job_entropy, mch_entropies
            )

            self.job_optimizer.zero_grad()
            job_loss_sum.backward(retain_graph=True)
            self.job_optimizer.step()

            self.mch_optimizer.zero_grad()
            mch_loss_sum.backward(retain_graph=True)
            self.mch_optimizer.step()

            # Update old policy and schedulers if needed
            self.policy_old_job.load_state_dict(self.policy_job.state_dict())
            self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
            if decay_flag:
                self.job_scheduler.step()
                self.mch_scheduler.step()

    def compute_loss(self, batch_size, job_log_prob, mch_log_prob, val, rewards_all_env, job_entropy, mch_entropies):
        job_loss_sum, mch_loss_sum = 0, 0
        for j in range(batch_size):
            job_ratios = torch.exp(job_log_prob[j] - job_log_prob[j].detach())
            mch_ratios = torch.exp(mch_log_prob[j] - mch_log_prob[j].detach())
            advantages = adv_normalize(rewards_all_env[j] - val[j].detach())

            job_surr1 = job_ratios * advantages
            job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            job_loss = (
                -torch.min(job_surr1, job_surr2) + 0.5 * self.MSE(val[j], rewards_all_env[j]) - 0.01 * job_entropy[j]
            )
            job_loss_sum += job_loss.mean()

            mch_surr1 = mch_ratios * advantages
            mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            mch_loss = -torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j]
            mch_loss_sum += mch_loss.mean()

        return job_loss_sum, mch_loss_sum


def main(epochs, params):
    # Initialize PPO, datasets, and data loaders
    ppo = PPO(**params)
    train_dataset = FJSPDataset(params["n_j"], params["n_m"], params["low"], params["high"], params["num_ins"], 200)
    validat_dataset = FJSPDataset(params["n_j"], params["n_m"], params["low"], params["high"], 128, 200)
    data_loader = DataLoader(train_dataset, batch_size=params["batch_size"])
    valid_loader = DataLoader(validat_dataset, batch_size=params["batch_size"])

    for epoch in range(epochs):
        memory = Memory()
        for batch_idx, batch in enumerate(data_loader):
            # Reset environment and process batch data
            ...
            # Perform training steps
            ppo.update(memory, params["batch_size"], params["decayflag"])
            memory.clear_memory()


# Set parameters and run main
params = {
    "lr": 0.0003,
    "gamma": 0.99,
    "k_epochs": 4,
    "eps_clip": 0.2,
    "decay_step_size": 100,
    "decay_ratio": 0.96,
    "n_j": 10,
    "n_m": 5,
    "num_layers": 3,
    "input_dim": 16,
    "hidden_dim": 64,
    "num_mlp_layers_feature_extract": 2,
    "num_mlp_layers_critic": 2,
    "hidden_dim_critic": 64,
    "vloss_coef": 0.5,
    "ploss_coef": 1.0,
    "entloss_coef": 0.01,
    "device": "cpu",
    "batch_size": 32,
    "decayflag": True,
    "low": 0,
    "high": 100,
    "num_ins": 200,
}

if __name__ == "__main__":
    main(1, params)
