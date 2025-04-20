
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
if not hasattr(np, "bool8"):
    np.bool8 = bool
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

class RA_PPO_Agent:
    def __init__(self, train_env, record_env, n_envs=4, batch_steps=2048,
                 clip_eps=0.2, lr=3e-4, gamma=0.99, lam=0.95, risk_alpha=0.05, beta=-5.0):
        self.train_env = train_env
        self.record_env = record_env
        self.n_envs = n_envs
        self.batch_steps = batch_steps
        obs_dim = train_env.single_observation_space.shape[0]
        act_dim = train_env.single_action_space.n
        self.model = PolicyNet(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma, self.lam = gamma, lam
        self.clip_eps, self.risk_alpha, self.beta = clip_eps, risk_alpha, beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.all_returns, self.returns_hist, self.cvar_hist = [], [], []

    def get_batch(self):
        obs, _ = self.train_env.reset()
        mb_obs, mb_acts, mb_logps, mb_vals, mb_rews, mb_dones = [], [], [], [], [], []
        for _ in range(self.batch_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            logits, vals = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)
            next_obs, rews, dones, _, _ = self.train_env.step(acts.cpu().numpy())
            mb_obs.append(obs)
            mb_acts.append(acts.cpu().numpy())
            mb_logps.append(logps.detach().cpu().numpy())
            mb_vals.append(vals.detach().cpu().numpy())
            mb_rews.append(rews)
            mb_dones.append(dones)
            obs = next_obs
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        _, last_vals = self.model(obs_tensor)
        return (np.array(mb_obs), np.array(mb_acts), np.array(mb_logps),
                np.array(mb_vals), np.array(mb_rews), np.array(mb_dones),
                last_vals.detach().cpu().numpy())

    def compute_gae(self, rews, vals, dones, last_vals):
        T,N = rews.shape
        returns = np.zeros((T,N), dtype=np.float32)
        advs    = np.zeros((T,N), dtype=np.float32)
        for n in range(N):
            gae=0
            for t in reversed(range(T)):
                mask = 1.0 - dones[t,n]
                delta = rews[t,n] + self.gamma*last_vals[n]*mask - vals[t,n]
                gae = delta + self.gamma*self.lam*mask*gae
                advs[t,n] = gae
                returns[t,n] = gae + vals[t,n]
                last_vals[n]=vals[t,n]
        return returns.flatten(), advs.flatten()

    def update(self, batch):
        mb_obs,mb_acts,mb_logps,mb_vals,mb_rews,mb_dones,last_vals = batch
        rets, advs = self.compute_gae(mb_rews,mb_vals,mb_dones,last_vals)
        obs = mb_obs.reshape(-1, mb_obs.shape[-1])
        acts = mb_acts.flatten()
        old_lp = mb_logps.flatten()
        dataset = list(zip(obs, acts, old_lp, rets, advs))
        np.random.shuffle(dataset)
        for _ in range(4):
            for start in range(0, len(dataset), 256):
                batch_slice = dataset[start:start+256]
                b_obs, b_acts, b_lp, b_rets, b_advs = zip(*batch_slice)
                b_obs = torch.FloatTensor(b_obs).to(self.device)
                b_acts= torch.LongTensor(b_acts).to(self.device)
                b_lp  = torch.FloatTensor(b_lp).to(self.device)
                b_rets= torch.FloatTensor(b_rets).to(self.device)
                b_advs= torch.FloatTensor(b_advs).to(self.device)
                b_advs=(b_advs-b_advs.mean())/(b_advs.std()+1e-8)
                logits, vals = self.model(b_obs)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(b_acts)
                ratio = torch.exp(lp-b_lp)
                s1=ratio*b_advs; s2=torch.clamp(ratio,1-self.clip_eps,1+self.clip_eps)*b_advs
                p_loss=-torch.min(s1,s2).mean()
                v_loss=(b_rets-vals).pow(2).mean()
                ent=dist.entropy().mean()
                loss = p_loss + 0.5*v_loss - 0.01*ent
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

    def train(self, iters=125):
        for i in range(iters):
            batch=self.get_batch()
            ep_returns=batch[4].sum(axis=0)
            self.all_returns.extend(ep_returns.tolist())
            self.returns_hist.extend(ep_returns.tolist())
            thr = np.quantile(self.returns_hist[-100:], self.risk_alpha) if len(self.returns_hist)>=100 else np.nan
            self.cvar_hist.append(thr)
            self.update(batch)
            print(f"Iter {i+1}/{iters}, AvgRet {ep_returns.mean():.2f}, CVaR {thr:.2f}")
        # record video
        obs=self.record_env.reset()
        done=False
        while not done:
            a,_= self.get_batch()[1][0::self.n_envs], None
            obs,_,done,_,_=self.record_env.step(int(a))
        self.record_env.close()
