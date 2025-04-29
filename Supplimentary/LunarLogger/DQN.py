import os, time, random, collections, csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import math

# ----------------- CONFIG -----------------
ENV_ID              = "LunarLander-v3"
TOTAL_EPISODES      = 1000
MAX_STEPS_PER_EP    = 1000
BUFFER_CAP          = 100_000
INITIAL_MEMORY      = 10_000
BATCH_SIZE          = 64
GAMMA               = 0.99
LR                  = 5e-4
HIDDEN_DIM          = 256
UPDATE_EVERY        = 4
TARGET_UPDATE_FREQ  = 1000
EPS_START           = 1.0
EPS_END             = 0.01
EPS_DECAY           = 0.995
LOG_DIR             = "logs/dqn_logged"
CSV_LOG             = os.path.join(LOG_DIR, "dqn_metrics.csv")
PLOTS_DIR           = LOG_DIR
VIDEO_DIR           = os.path.join(LOG_DIR, "videos")
# -------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        )
    def forward(self, x):
        return self.net(x)

Experience = collections.namedtuple("Experience", ["s","a","r","s2","d"])
class ReplayBuffer:
    def __init__(self, cap): self.buf = collections.deque(maxlen=cap)
    def add(self, *args): self.buf.append(Experience(*args))
    def sample(self, k):
        ex = random.sample(self.buf, k)
        s  = torch.tensor([e.s for e in ex], dtype=torch.float32).to(device)
        a  = torch.tensor([e.a for e in ex], dtype=torch.int64).to(device)
        r  = torch.tensor([e.r for e in ex], dtype=torch.float32).to(device)
        s2 = torch.tensor([e.s2 for e in ex], dtype=torch.float32).to(device)
        d  = torch.tensor([e.d for e in ex], dtype=torch.float32).to(device)
        return s,a,r,s2,d
    def __len__(self): return len(self.buf)

class DQNAgent:
    def __init__(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        # Networks
        self.env = gym.make(ENV_ID)
        self.qnet = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.target = QNetwork(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=LR)
        self.buf = ReplayBuffer(BUFFER_CAP)
        self.loss_fn = nn.SmoothL1Loss()
        # Logging
        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()
        self.returns = []
        self.losses = []

        os.makedirs(LOG_DIR, exist_ok=True)

        # Auto-increment run number
        existing_runs = glob.glob(os.path.join(LOG_DIR, 'run*.csv'))
        run_id = len(existing_runs) + 1
        run_csv_path = os.path.join(LOG_DIR, f'run{run_id}.csv')


        os.makedirs(LOG_DIR, exist_ok=True)

        existing_runs = glob.glob(os.path.join(LOG_DIR, 'run*.csv'))
        run_id = len(existing_runs) + 1
        run_csv_path = os.path.join(LOG_DIR, f'run{run_id}.csv')

        self.csv_file = open(run_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'iter', 'env_steps', 'elapsed_time_s', 'steps_per_s', 'computation_speed_sps',
            'avg_return', 'cvar', 'policy_loss', 'value_loss', 'entropy', 'avg_loss'
        ])

    def fill_buffer(self):
        print("Filling replay buffer...")
        obs, _ = self.env.reset()
        for _ in range(INITIAL_MEMORY):
            action = self.env.action_space.sample()
            nxt, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.buf.add(obs, action, reward, nxt, float(done))
            obs = nxt if not done else self.env.reset()[0]
        print("Replay buffer ready.")

    def train(self):
        self.fill_buffer()
        epsilon = EPS_START
        rolling = []
        for epi in range(1, TOTAL_EPISODES+1):
            obs, _ = self.env.reset()
            total_r = 0.0
            loss_accum = 0.0
            loss_count = 0
            for t in range(MAX_STEPS_PER_EP):
                # ε-greedy
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        qs = self.qnet(torch.tensor(obs, dtype=torch.float32).to(device))
                    action = qs.argmax().item()
                nxt, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                self.buf.add(obs, action, reward, nxt, float(done))
                obs = nxt; total_r += reward
                # learning update
                if len(self.buf) >= BATCH_SIZE and self.step_count % UPDATE_EVERY == 0:
                    s,a,r,s2,d = self.buf.sample(BATCH_SIZE)
                    q_pred = self.qnet(s).gather(1,a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = self.qnet(s2).argmax(dim=1,keepdim=True)
                        q_next = self.target(s2).gather(1,next_actions).squeeze(1)
                    q_target = r + GAMMA * q_next * (1-d)
                    loss = self.loss_fn(q_pred, q_target)
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.qnet.parameters(),1.0)
                    self.opt.step()
                    loss_accum += loss.item(); loss_count += 1
                # target sync
                if self.step_count % TARGET_UPDATE_FREQ == 0:
                    self.target.load_state_dict(self.qnet.state_dict())
                self.step_count += 1
                if done: break
            # end episode
            self.episode_count += 1
            self.returns.append(total_r)
            avg_loss = loss_accum/loss_count if loss_count>0 else 0.0
            rolling.append(np.mean(self.returns[-20:]))
            elapsed = time.time() - self.start_time
            speed = self.step_count / elapsed if elapsed>0 else 0.0
            # log CSV
            self.csv_writer.writerow([
                epi, self.step_count, elapsed, speed,
                speed,
                total_r, math.nan,  # no CVaR
                math.nan, math.nan, math.nan,  # no policy_loss, value_loss, entropy
                avg_loss
            ])

            if epi % 20 == 0:
                print(f"[DQN] Ep {epi}/{TOTAL_EPISODES} | Ret {total_r:.2f} | "
                      f"20-ep avg {rolling[-1]:.2f} | eps {epsilon:.3f}")
            epsilon = max(EPS_END, EPS_DECAY * epsilon)
        # finish logging
        self.csv_file.close()
        # plots
        steps = np.arange(1,TOTAL_EPISODES+1)*MAX_STEPS_PER_EP
        plt.figure()
        plt.plot(steps, self.returns, label='Return')
        plt.plot(steps, rolling, label='20-ep rolling')
        plt.xlabel('Env Steps'); plt.ylabel('Return'); plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR,'dqn_return_vs_steps.png'))
        plt.figure()
        plt.plot(steps, np.linspace(speed,len(steps)*speed,len(steps)),label='Speed(s/s)')
        plt.xlabel('Env Steps'); plt.ylabel('Speed'); plt.savefig(os.path.join(PLOTS_DIR,'dqn_speed_vs_steps.png'))
        # final rollout
        print("🎥 Recording final rollout...")
        vid_env = RecordVideo(gym.make(ENV_ID, render_mode='rgb_array'),video_folder=VIDEO_DIR,name_prefix='dqn_rollout')
        obs,_=vid_env.reset(); done=False; tot=0; cnt=0
        while not done and cnt<MAX_STEPS_PER_EP:
            with torch.no_grad(): action=self.qnet(torch.tensor(obs,dtype=torch.float32).to(device)).argmax().item()
            obs,rew,term,tr, _ = vid_env.step(action); done=term or tr; tot+=rew; cnt+=1
        vid_env.close(); print(f"✅ Rollout: steps={cnt}, reward={tot:.2f}")

if __name__ == '__main__':
    agent = DQNAgent()
    agent.train()
