

import gymnasium as gym
from gymnasium import spaces 
import numpy as np
import math
from operator import attrgetter
from collections import namedtuple
import random

Agent = namedtuple("Agent", ["y", "x", "type_id"])

class WildFireEnv(gym.Env):
    def __init__(self, n_grid = 3, method = "baseline", mode = 'train'):
        super(WildFireEnv, self).__init__()

        self.n_grid = n_grid
        self.method = method

        self.grid_size = (self.n_grid, self.n_grid) 
        self.FF = [[2, 0], [3, 0]]
        self.med = [[1, 0], [1, 0]]
        self.original_FF = self.FF
        self.original_med = self.med
        self.number_of_FF = len(self.FF)
        self.number_of_med = len(self.med)
        self.FF_id = 1000
        self.med_id = 2000
        self.fire_id = 3000
        self.victim_id = 4000
        self.agents = {}
        self.n_agents = 0
        self.fire = [[0, 1], [1, 2], [2, 1]]
        self.victims = [[0, 0], [1, 2]]
        self.n_objects = len(self.fire) + len(self.victims)
        self.original_fire = self.fire
        self.original_victims = self.victims
        self.victim_saved = 0
        self.fire_ex = 0
        self.trajectory = list()
        if mode == 'train':
            self.max_step = 1000
        else:
            self.max_step = 30000

        self.init_agents()

        self.mode = mode
        self.trunct = False

        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)

        

    def init_agents(self):
        ally_agents = []

        for ff in self.FF:
            ally_agents.append(Agent(ff[0], ff[1], self.FF_id)) 
        
        for med in self.med:
            ally_agents.append(Agent(med[0], med[1], self.med_id))

        sorted_ally_agents = sorted(
            ally_agents, 
            key=attrgetter("x", "y", "type_id"),
            reverse = False           
            )
        
        for i in range(len(sorted_ally_agents)):
            self.agents[i] = sorted_ally_agents[i]
        
        self.n_agents = len(sorted_ally_agents)
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents)

    def get_unit_by_id(self, agent_id):
        return self.agents[agent_id]

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id = None):
        """Returns observation for agent_id. The observation is composed of:

        - agent features (position in the path and agents id)
        - enemy features (available_to_see, distance relative_x, relative_y,
            , unit_type)
        - objects features (visible, distance, relative_x, relative_y,
            unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """

        if agent_id == None:
            raise(f"Agent ID is not selected")

        unit = self.get_unit_by_id(agent_id)

        agent_feats = np.zeros((self.n_agents-1, 5), dtype = np.int64)
        object_feats = np.zeros((len(self.original_fire) + len(self.original_victims), 5), dtype = np.int64)
        own_feats = np.zeros(3, dtype = np.int64)

        x = unit.x
        y = unit.y

        # object features
        i = 0
        for f in self.fire:
            fy, fx = f
            dist = self._manhattan_distance(f, (y, x))

            if (dist <= 1):
                object_feats[i, 0] = 1 # visible
                relative_x = (fx - x)
                relative_y = (fy - y)

                object_feats[i, 1] = relative_x # relative x
                object_feats[i, 2] = relative_y # relative y
                object_feats[i, 3] = dist # distance
                object_feats[i, 4] = self.fire_id # id for fire
            i += 1
        
        for v in self.victims:
            vy, vx = v
            dist = self._manhattan_distance(v, (y, x))

            if (dist <= 1):
                object_feats[i, 0] = 1 # visible
                relative_x = (vx - x)
                relative_y = (vy - y)

                object_feats[i, 1] = relative_x # relative x
                object_feats[i, 2] = relative_y # relative y
                object_feats[i, 3] = dist # distance
                object_feats[i, 4] = self.victim_id # id for victim
            i += 1
        
        # agent features
        ally_ids = [id for id in range(self.n_agents) if id != agent_id]


        for i, ally_id in enumerate(ally_ids):
            ally_unit = self.get_unit_by_id(ally_id)
            ax = ally_unit.x
            ay = ally_unit.y
            dist = self._manhattan_distance((ax, ay), (x, y))

            if (dist <= 1):
                agent_feats[i, 0] = 1 # visible
                relative_x = (ax - x)
                relative_y = (ay - y)

                agent_feats[i, 1] = relative_x
                agent_feats[i, 2] = relative_y
                agent_feats[i, 3] = dist

                # something
                agent_feats[i, 4] = ally_unit.type_id


        #own feats
        own_feats[0] = unit.x
        own_feats[1] = unit.y
        own_feats[2] = unit.type_id

        agent_obs = np.concatenate((
            own_feats.flatten(), 
            agent_feats.flatten(),
            object_feats.flatten()
            )
        )

        np.round(agent_obs, decimals=0, out=None)

        return agent_obs
    
    def get_obs_size(self):
        """Returns the size of the observation."""
        tot_objects = self.n_objects * 5
        tot_agnets = (self.n_agents - 1) * 5
        itself = 3

        return tot_objects + tot_agnets + itself
    
    def get_grid(self):
        grid = np.zeros((self.n_grid, self.n_grid), dtype = np.int8)

        cells = {}

        # Fires
        for f in self.fire:
            cells.setdefault(tuple(f), set()).add("f")

        # Victims
        for v in self.victims:
            cells.setdefault(tuple(v), set()).add("v")
        
        for agent in self.agents.values():
            a_coords = (agent.y, agent.x)
            type = "FF" if agent.type_id == 1000 else "med"
            cells.setdefault(a_coords, set()).add(type)

        for coords, content in cells.items():
            if content == {'FF'}:
                grid[coords] = 1
            elif content == {'med'}:
                grid[coords] = 2
            elif content == {'med', 'FF'}:
                grid[coords] = 3
            elif content == {'f'}:
                grid[coords] = 4
            elif content == {'v'}:
                grid[coords] = 5
            elif content == {'FF', 'f'}:
                grid[coords] = 6
            elif content == {'FF', 'v'}:
                grid[coords] = 7
            elif content == {'v', 'f'}:
                grid[coords] = 8
            elif content == {'med', 'f'}:
                grid[coords] = 9
            elif content == {'med', 'v'}:
                grid[coords] = 10
            elif content == {'med', 'FF', 'v'}:
                grid[coords] = 11
            elif content == {'med', 'FF', 'f'}:
                grid[coords] = 12
            elif content == {'med', 'f', 'v'}:
                grid[coords] = 13
            elif content == {'FF', 'f', 'v'}:
                grid[coords] = 14
            elif content == {'FF', 'f', 'v', 'med'}:
                grid[coords] = 15

        return grid
    
    def get_state(self):
        agent_feats = np.zeros((len(self.FF) + len(self.med), 3), dtype = np.long)
        object_feats = np.zeros((len(self.original_fire) + len(self.original_victims), 3), dtype = np.long)
        
        i = 0

        for f in self.fire:
            fy, fx = f
            object_feats[i, 0] = fx
            object_feats[i, 1] = fy
            object_feats[i, 2] = self.fire_id

            i += 1
        
        for v in self.victims:
            vy, vx = v
            object_feats[i, 0] = vx
            object_feats[i, 1] = vy
            object_feats[i, 2] = self.victim_id

            i += 1
        
        # agent features
        ids = [id for id in range(self.n_agents)]

        for i, id in enumerate(ids):
            agent = self.get_unit_by_id(id)
            ax = agent.x
            ay = agent.y

            agent_feats[i, 0] = ax
            agent_feats[i, 1] = ay
            agent_feats[i, 2] = agent.type_id

        
        state = np.concatenate((
            agent_feats.flatten(),
            object_feats.flatten()
            )
        )

        return state

    def get_state_size(self):
        """Returns the size of the global state."""
        # if self.obs_instead_of_state:
        #     return self.get_obs_size() * self.n_agents

        object_state = self.n_objects * 3
        agent_state = self.n_agents * 3

        size = object_state + agent_state

        # if self.state_last_action:
        #     size += self.n_agents * self.n_actions
        # if self.state_timestep_number:
        #     size += 1
        # TODO

        return size



    def transition(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dy, dx) - Up, Down, Left, Right, Stay
        transition_probabilities = {
            0: (0.9, 0.025, 0.025, 0.025, 0.025),  # up
            1: (0.025, 0.9, 0.025, 0.025, 0.025),  # down
            2: (0.025, 0.025, 0.9, 0.025, 0.025),  # left
            3: (0.025, 0.025, 0.025, 0.9, 0.025),  # right
            4: (0.025, 0.025, 0.025, 0.025, 0.9)   # stay
        }

        probablities = transition_probabilities[action]
        

        #print("CHOICE", random.choices(moves, weights = probablities, k = 1)[0])

        dy, dx = random.choices(moves, weights = probablities, k = 1)[0]

        return dy, dx

    def step(self, actions):
        # print('action',action)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dy, dx) - Up, Down, Left, Right, Stay
        
        # print("FF", self.FF)
        # print("med", self.med)

        if self.mode == 'inference':
            for agent_id, action in enumerate(actions):
                agent = self.agents[agent_id]
                
                dy, dx = self.transition(action)

                new_agent = Agent(agent.x + dx, agent.y + dy, agent.type_id)
                self.agents[agent_id] = new_agent
            #new_FF = [self.FF[0] + moves[int(act[0])][0], self.FF[1] + moves[int(act[0])][1]]
            #new_med = [self.med[0] + moves[int(act[1])][0], self.med[1] + moves[int(act[1])][1]]
        else:
            for agent_id, action in enumerate(actions):
                agent = self.agents[agent_id]
                
                dy, dx = self.transition(action)

                new_agent = Agent(agent.x + dx, agent.y + dy, agent.type_id)
                self.agents[agent_id] = new_agent
            
            #new_FF = [self.FF[0] + moves[int(action[0])][0], self.FF[1] + moves[int(action[0])][1]]
            #new_med = [self.med[0] + moves[int(action[1])][0], self.med[1] + moves[int(action[1])][1]]


        # Move agents
        

        # print("new_FF", new_FF)
        # print("new_med", new_med)

        for agent_id, agent in self.agents.items():
            x = np.clip(agent.x, 0, self.n_grid - 1)
            y = np.clip(agent.y, 0, self.n_grid - 1)
            self.agents[agent_id] = Agent(x, y, agent.type_id)


        # print("new_FF1", self.FF)
        # print("new_med1", self.med)
        self.trajectory.append((self.FF, self.med, self.calculate_distance_med_FF()))

        reward = self.reward() 

        agent_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        #state = np.array(agent_obs).flatten()
        state = self.get_state()

        vistm_copy = self.victims.copy()
        fire_copy = self.fire.copy()

        for id in enumerate(self.agents):
            agent = self.agents[id[0]]
            a_coords = [agent.y, agent.x]

            if (a_coords in fire_copy) and (agent.type_id == self.FF_id):
                self.fire_ex += 1
                fire_copy.remove(a_coords)# Extinguish fire
            
            if a_coords in vistm_copy and (agent.type_id == self.med_id):
                self.victim_saved += 1
                vistm_copy.remove(a_coords) #save victim

        self.victims = vistm_copy.copy()
        self.fire = fire_copy.copy()

        terminated = len(self.fire) == 0 and len(self.victims) == 0
        sub_goals = [len(self.fire) == 0 , len(self.victims) == 0]

        info = {
        "fires_extinguished": self.fire_ex,
        "victims_saved": self.victim_saved,
        "sub_goals": sub_goals}

        
        if len(self.trajectory) > self.max_step:
            terminated = True
            self.trunct =True

        return agent_obs, state, reward, terminated, self.trunct, info
    
    def calculate_distance_med_FF(self):
        return 0
        return abs(self.FF[0] - self.med[0]) + abs(self.FF[1] - self.med[1])

    def calculate_distance(self, start, target):
        return abs(start[0] - target[0]) + abs(start[1] - target[1])
    
    def reward(self):
            return 5
        
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.FF = self.original_FF
        self.med = self.original_med

        self.fire = [[0, self.n_grid -1], [3, self.n_grid -1], [4, self.n_grid -1]]  
        self.victims = [[0, 0], [0, self.n_grid -1]]
        self.original_fire = self.fire
        self.original_victims = self.victims
        self.victim_saved = 0
        self.fire_ex = 0 
        self.trunct = False
        self.trajectory = list()
        self.trajectory.append((self.FF, self.med, self.calculate_distance_med_FF()))
        agent_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        #state = np.array(agent_obs).flatten()
        state = self.get_state()
        self.init_agents()
        self.action_space = spaces.MultiDiscrete([5] * self.n_agents) 

        return agent_obs, state, {}

    def render(self):
        grid = np.full((self.n_grid, self.n_grid), ' . ', dtype=object)  

        cells = {}


        # Fires
        for f in self.fire:
            cells.setdefault(tuple(f), set()).add("f")

        # Victims
        for v in self.victims:
            cells.setdefault(tuple(v), set()).add("v")
        
        for agent in self.agents.values():
            a_coords = (agent.y, agent.x)
            type = "FF" if agent.type_id == 1000 else "med"
            cells.setdefault(a_coords, set()).add(type)

            #grid[tuple(v)] = 8 if v in self.fire else 5

        for coords, content in cells.items():
            if content == {'FF'}:
                grid[coords] = 'FF'
            elif content == {'med'}:
                grid[coords] = 'MD'
            elif content == {'med', 'FF'}:
                grid[coords] = 'FM'
            elif content == {'f'}:
                grid[coords] = '🔥'
            elif content == {'v'}:
                grid[coords] = 'V'
            elif content == {'FF', 'f'}:
                grid[coords] = 'FF🔥'
            elif content == {'FF', 'v'}:
                grid[coords] = 'FFV'
            elif content == {'v', 'f'}:
                grid[coords] = 'V🔥'
            elif content == {'med', 'f'}:
                grid[coords] = 'MD🔥'
            elif content == {'med', 'v'}:
                grid[coords] = 'MDV'
            elif content == {'med', 'FF', 'v'}:
                grid[coords] = 'FMV'
            elif content == {'med', 'FF', 'f'}:
                grid[coords] = 'FM🔥'
            elif content == {'med', 'f', 'v'}:
                grid[coords] = 'MV🔥'
            elif content == {'FF', 'f', 'v'}: 
                grid[coords] = 'FV🔥'
            elif content == {'med', 'FF', 'f', 'v'}: 
                grid[coords] = 'FMV🔥'

        temp_victim = self.victims.copy()

        formatted_grid = "\n".join(["  ".join(f"{cell:3}" for cell in row) for row in grid])
        print('###################\n\n\n###################')
        print(formatted_grid)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

if __name__ == "__main__":

    env = WildFireEnv(method="hypRL", n_grid=5)


    # env.reset()
    # env.render()

    # done = False
    # step = 0

    # print(env.observation_space.sample())
    # print(env.observation_space)
    # for i in range(10):        
    #     action = env.action_space.sample()
        
    #     obs, state, reward, done, trunct, info = env.step(action)
    #     env.render()
    #     print("STARTING STATE")
    #     print(state)

    #     step += 1 
    #     print("reward", reward)



