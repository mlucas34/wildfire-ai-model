from torch.utils.data import Dataset
import torch
import numpy as np

class WildFireDataset(Dataset):
    def __init__(self, env, num_episodes=100, steps=20, agent_idx=0):
        self.env = env
        self.num_episodes = num_episodes
        self.steps = steps
        self.agent_idx = agent_idx
        self.samples = []
        
    def collect_data(self):
        for _ in range(self.num_episodes):
            obs, state, _ = self.env.reset()
            cur_obs = self.env.get_agent_obs(self.agent_idx)

            state = np.array(state.flatten(), dtype=np.long)
            self.samples.append((np.array(cur_obs, dtype=np.long), state))

            for i in range(self.steps):        
                action = self.env.action_space.sample()
                
                obs, state, reward, done, trunct, info = self.env.step(action)
                cur_obs = self.env.get_agent_obs(self.agent_idx)

                if done:
                    break

                state = np.array(state.flatten(), dtype=np.long)
                self.samples.append((np.array(cur_obs, dtype=np.long), state))

                #step += 1 
                #print("reward", reward)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src, trg = self.samples[idx]
        return torch.tensor(src, dtype=torch.float32), torch.tensor(trg.flatten(), dtype=torch.long)



