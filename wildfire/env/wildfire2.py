import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from operator import attrgetter
from collections import namedtuple
import random
<<<<<<< HEAD
#from smac.env.multiagentenv import MultiAgentEnv
=======
from smac.env.multiagentenv import MultiAgentEnv
>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7
from enum import Enum

Agent = namedtuple("Agent", ["x", "y", "type_id", 'agent_id'])
obj = namedtuple("obj", ["x", "y", "type_id", 'obj_id'])


class Moves(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

<<<<<<< HEAD
class WildFireEnv():
=======
class WildFireEnv(MultiAgentEnv):
>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7
    def __init__(self, n_grid = 3, hyper = False, seed = None, episode_limit = 50):
        super(WildFireEnv, self).__init__()

        # info
        self.n_grid = n_grid
        self.hyper = hyper
        self.grid_size = (self.n_grid, self.n_grid)
        self.episode_limit = episode_limit
        self.safe_dist = 5
        self.max_dist_x = n_grid
        self.max_dist_y = n_grid
        self.sight_range = 5

        # actions
        self.n_actions = 4

        # objects
        fire = [[0, 2]]
        victim = [[1, 2]]

        self.fire = fire.copy()
        self.victims = victim.copy()

        self.fire_org = fire.copy()
        self.victims_org = victim.copy()


        self.n_objects = len(self.fire_org) + len(self.victims_org)
        self.fire_id = 1
        self.victim_id = 2
        self.original_fire = self.fire
        self.original_victims = self.victims

        # agents
        self.FF = [[2, 0]]
        self.med = [[2, 0]]
        self.FF_id = 4
        self.med_id = 5
        max_id_num = max(self.fire_id, self.victim_id, self.FF_id, self.med_id)
        self.max_bits = len(list(bin(max_id_num)[2:]))
        self.agents = {}
        self.objects = {}
        self.n_agents = len(self.FF) + len(self.med)

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # goals
        self.victim_saved = 0
        self.fire_ex = 0
        self.trajectory = list()

        self.init_agents()
        self.init_objects()



        self.steps = 0

        self._seed = seed

        self.init_reward()

    def init_agents(self):
        ally_agents = []

        for index_ff, ff in enumerate(self.FF):
            ally_agents.append(Agent(ff[0], ff[1], self.FF_id, self.FF_id)) 
        
        for index_med, med in enumerate(self.med):
            ally_agents.append(Agent(med[0], med[1], self.med_id, self.med_id))

        sorted_ally_agents = sorted(
            ally_agents, 
            key=attrgetter("x", "y", "type_id", "agent_id"),
            reverse = False           
            )
        
        for i in range(len(sorted_ally_agents)):
            self.agents[i] = sorted_ally_agents[i]

    def init_objects(self):
        objects = []

        for index_fire, fire in enumerate(self.fire_org):
            objects.append(obj(fire[0], fire[1], self.fire_id, self.fire_id)) 
        
        for index_vic, victim in enumerate(self.victims_org):
            objects.append(obj(victim[0], victim[1], self.victim_id, self.victim_id))

        sorted_objects = sorted(
            objects, 
            key=attrgetter("x", "y", "type_id", "obj_id"),
            reverse = False           
            )
        
        for i in range(len(sorted_objects)):
            self.objects[i] = sorted_objects[i]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        self.fire = [p.copy() for p in self.fire_org]
        self.victims = [p.copy() for p in self.victims_org]
        self.agents.clear()
        self.init_agents()
        self.objects.clear()
        self.init_objects()
        self.steps = 0
        self.victim_saved = 0
        self.fire_ex = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        self.init_reward()
        return

    def get_unit_by_id(self, agent_id):
        return self.agents[agent_id]
    
    def get_obj_unit_by_id(self, obj_id):
        return self.objects[obj_id]
    
    def padded(self, id):
        a = np.array(list(bin(id)[2:]))
        padded = np.pad(a, (max(0,self.max_bits - len(a)), 0), mode='constant')
        return padded

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
        - Last action of the agents 

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """

        if agent_id == None:
            raise(f"Agent ID is not selected")

        unit = self.get_unit_by_id(agent_id)

        agent_feats = np.zeros((self.n_agents-1, 4 + self.max_bits), dtype = np.float32)
        object_feats = np.zeros((self.n_objects, 4 + self.max_bits), dtype = np.float32)
        own_feats = np.zeros(2+self.max_bits , dtype = np.float32)
        move_feats = np.zeros(self.n_actions, dtype=np.float32)

        x = unit.x
        y = unit.y

        # object features

        obj_ids = [id for id in range(self.n_objects)]

        for i, obj_id in enumerate(obj_ids):
            obj_unit = self.get_obj_unit_by_id(obj_id)
            ax = obj_unit.x
            ay = obj_unit.y
            dist = self._manhattan_distance((ax, ay), (x, y))

            if (dist <= self.sight_range):
                object_feats[i, 0] = 1 # visible
<<<<<<< HEAD
                relative_x = (x - ax)
                relative_y = (ay - y)

                object_feats[i, 1] = relative_y/self.max_dist_x
                object_feats[i, 2] = relative_x/self.max_dist_y
=======
                relative_x = (ax - x)
                relative_y = (ay - y)

                object_feats[i, 1] = relative_x/self.max_dist_x
                object_feats[i, 2] = relative_y/self.max_dist_y
>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7
                object_feats[i, 3] = dist/self.sight_range
                object_feats[i, 4:4+self.max_bits] = self.padded(obj_unit.obj_id)

        
        # agent features
        ally_ids = [id for id in range(self.n_agents) if id != agent_id]


        for i, ally_id in enumerate(ally_ids):
            ally_unit = self.get_unit_by_id(ally_id)
            ax = ally_unit.x
            ay = ally_unit.y
            dist = self._manhattan_distance((ax, ay), (x, y))

            if (dist <= self.sight_range):
                agent_feats[i, 0] = 1 # visible
                relative_x = (x - ax)
                relative_y = (ay - y)

<<<<<<< HEAD
                agent_feats[i, 1] = relative_y/self.max_dist_x
                agent_feats[i, 2] = relative_x/self.max_dist_y
=======
                agent_feats[i, 1] = relative_x/self.max_dist_x
                agent_feats[i, 2] = relative_y/self.max_dist_y
>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7
                agent_feats[i, 3] = dist/self.sight_range

                # something
                # agent_feats[i, 4] = ally_unit.type_id
                agent_feats[i, 4:4+self.max_bits] = self.padded(ally_unit.agent_id)

        # movemnet features
        avail_actions = self.get_avail_agent_actions(agent_id)
        for m in range(self.n_actions):
                move_feats[m] = avail_actions[m] 


        #own feats
        own_feats[0] = unit.x
        own_feats[1] = unit.y
        own_feats[2:] =  self.padded(unit.agent_id)

        

        agent_obs = np.concatenate((
            own_feats.flatten(), 
            agent_feats.flatten(),
            object_feats.flatten(),
            move_feats.flatten(),
            self.last_action[agent_id].flatten()
            )
        )

        agent_obs = agent_obs.astype(dtype=np.float32)

        return agent_obs
    
    def get_obs_size(self):
        """Returns the size of the observation."""
        tot_objects = self.n_objects * (4 + self.max_bits)
        tot_agnets = (self.n_agents - 1) * (4 + self.max_bits)
        itself = 2 + self.max_bits
        movement = self.n_actions
        last_action = self.n_actions

        return tot_objects + tot_agnets + itself + movement + last_action
    
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
            a_coords = (agent.x, agent.y)
            type = "FF" if agent.type_id == self.FF_id else "med"
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
        agent_feats = np.zeros((self.n_agents, 2+self.max_bits), dtype = np.float32)
        object_feats = np.zeros((self.n_objects, 2+self.max_bits), dtype = np.float32)
        

        ids = [id for id in range(self.n_objects)]

        for i, id in enumerate(ids):
            obj = self.get_obj_unit_by_id(id)
            ax = obj.x
            ay = obj.y

            object_feats[i, 0] = ax
            object_feats[i, 1] = ay
            object_feats[i, 2:2+self.max_bits] = self.padded(obj.obj_id)
        
        # agent features
        ids = [id for id in range(self.n_agents)]

        for i, id in enumerate(ids):
            agent = self.get_unit_by_id(id)
            ax = agent.x
            ay = agent.y

            agent_feats[i, 0] = ax
            agent_feats[i, 1] = ay
            agent_feats[i, 2:2+self.max_bits] = self.padded(agent.agent_id)
        
        time_step = np.array([self.steps / self.episode_limit])

        state = np.concatenate((
            agent_feats.flatten(),
            object_feats.flatten(),
            self.last_action.flatten(),
            time_step.flatten()
            )
        )

        state = state.astype(dtype=np.float32)

        return state
    
    def close(self):
        pass

    def get_state_size(self):
        """Returns the size of the global state."""
        # if self.obs_instead_of_state:
        #     return self.get_obs_size() * self.n_agents

        object_state = self.n_objects * (2+ self.max_bits)
        agent_state = self.n_agents * (2+ self.max_bits)

        #LAST ACTION
        last_action = self.n_agents * self.n_actions

        time_step =1

        size = object_state + agent_state + last_action + time_step

        return size


    def transition(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dx, dy) - Up, Down, Left, Right, Stay
        # transition_probabilities = {
        # 0: (1.0, 0, 0, 0, 0),  # up
        # 1: (0, 1.0, 0, 0, 0),  # down
        # 2: (0, 0, 1.0, 0, 0),  # left
        # 3: (0, 0, 0, 1.0, 0),  # right
        # 4: (0, 0, 0, 0, 1.0)
        # }

        transition_probabilities = {
        0: (1.0, 0, 0, 0),  # up
        1: (0, 1.0, 0, 0),  # down
        2: (0, 0, 1.0, 0),  # left
        3: (0, 0, 0, 1.0),  # right
        }


        probablities = transition_probabilities[action]
        

        #print("CHOICE", random.choices(moves, weights = probablities, k = 1)[0])

        dx, dy = random.choices(moves, weights = probablities, k = 1)[0]

        return dx, dy

    def step(self, actions):

        terminated = False

        self.steps += 1


        actions = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions)]

        
        for agent_id, action in enumerate(actions):
            agent = self.agents[agent_id]
            
            dx, dy = self.transition(action)

            new_agent = Agent(agent.x + dx, agent.y + dy, agent.type_id, agent.agent_id)
            self.agents[agent_id] = new_agent
            

        # Move agents

        for agent_id, agent in self.agents.items():
            x = np.clip(agent.x, 0, self.n_grid - 1)
            y = np.clip(agent.y, 0, self.n_grid - 1)
            self.agents[agent_id] = Agent(x, y, agent.type_id,agent.agent_id)

        victim_copy = self.victims.copy()
        fire_copy = self.fire.copy()

        reward = 0

        for id in enumerate(self.agents):
            agent = self.agents[id[0]]

            a_coords = [agent.x, agent.y]


            if (a_coords in fire_copy) and (agent.type_id == self.FF_id):
                self.fire_ex += 1
                fire_copy.remove(a_coords)# Extinguish fire
                reward += 5
            
            if a_coords in victim_copy and (agent.type_id == self.med_id):
                self.victim_saved += 1
                victim_copy.remove(a_coords) #save victim
                reward+= 10

        self.victims = victim_copy.copy()
        self.fire = fire_copy.copy()

        #Ad Hoc reward
        for id in enumerate(self.agents):
            agent = self.agents[id[0]]

            a_coords = [agent.x, agent.y]

            if a_coords in self.fire and (agent.type_id == self.med_id):
                reward -= 1
            for id_2 in enumerate(self.agents):
                agent2 = self.agents[id_2[0]]
                if id == id_2:
                    continue
                else:
                    if self.distance(agent2.x, agent2.y, agent.x, agent.y)>self.safe_dist:
                        pass
        
        # self.render()

        info = self.get_env_info()
        

        info['Win'] = terminated = len(self.fire) == 0 and len(self.victims) == 0

        if info['Win'] :
            reward +=100

        
        # reward = self.get_reward()


        info['fires_extinguished'] = self.fire_ex
        info['victims_saved'] = self.victim_saved


        dist_agents = list()

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                ag1 = self.get_unit_by_id(i)
                ag2 = self.get_unit_by_id(j)

                dist_agents.append(self.distance(ag1.x, ag1.y, ag2.x, ag2.y))

        info['distance_agents'] = sum(dist_agents)/len(dist_agents)


        if self.steps >= self.episode_limit:
            reward = -10
            terminated = True

        reward = reward/10

        # print(reward)

        # self.render()

        return reward, terminated, info
    

        
    def seed(self):
        return self._seed

<<<<<<< HEAD
=======


>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7

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
            a_coords = (agent.x, agent.y)
            type = "FF" if agent.type_id == self.FF_id else "med"
            cells.setdefault(a_coords, set()).add(type)

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
        print('###################\n###################')
        print(formatted_grid)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
    
    def get_total_actions(self):
        return self.n_actions

    
    def init_reward(self):

        self.dist_fire = dict()

        self.dist_med = dict()


        for f_index, pos in enumerate(self.fire_org):
            self.dist_fire[f_index] = -1*math.inf

        for f_index, pos in enumerate(self.victims_org):
            self.dist_med[f_index] = -1*math.inf

        
        self.dist_med_ff = math.inf

        self.prev_victim = 0
        self.prev_fire = 0


    def get_reward(self):

        MULTI_WIN = 400

        MULTI_MED = 20
        MULTI_FF = 10
        MULTI_MED_FIRE = 50

        MULTI_DIST = 100

        MULTI_SAVE = 100
        MULTI_EX = 80

        # phi_save
        for agent_id in range(self.n_agents):
            ag = self.get_unit_by_id(agent_id)
            if ag.type_id == self.FF_id:
                for f_index, pos in enumerate(self.fire_org):
                    if pos not in self.fire:
                        self.dist_fire[f_index] = math.inf
                    else:
                        dist_temp = (1 - self.distance(pos[0],pos[1], ag.x, ag.y))*MULTI_FF
                        if dist_temp>self.dist_fire[f_index]:
                            self.dist_fire[f_index] = dist_temp
            phi_dist_fire = min(self.dist_fire.values())

            if ag.type_id == self.med_id:
                for f_index, pos in enumerate(self.victims_org):
                    if pos not in self.victims:
                        self.dist_med[f_index] = math.inf
                    else:
                        dist_temp = (1 - self.distance(pos[0],pos[1], ag.x, ag.y))* MULTI_MED
                        if dist_temp>self.dist_med[f_index]:
                            self.dist_med[f_index] = dist_temp

            phi_dist_victim = min(self.dist_med.values())


            temp_dist_med = list()


            if ag.type_id == self.med_id:
                for f_index, pos in enumerate(self.fire):
                    temp_dist_med.append((self.distance(pos[0],pos[1], ag.x, ag.y)) * MULTI_MED_FIRE)



            if ag.type_id == self.med_id and len(temp_dist_med) == 0:
                phi_dist_med_fire = math.inf
            elif ag.type_id == self.med_id and len(temp_dist_med) > 0:
                phi_dist_med_fire = min(temp_dist_med)

                

            temp_dist_med_ff = list()

            for agent_id_2 in range(self.n_agents):
                if agent_id_2 == agent_id:
                    continue
                ag2 = self.get_unit_by_id(agent_id_2)
                temp_dist_med_ff.append((self.safe_dist - self.distance(ag2.x,ag2.y, ag.x, ag.y))*MULTI_DIST )
            
            phi_dist_med_ff = min(temp_dist_med_ff)



        win = len(self.fire) == 0 and len(self.victims) == 0

        if win:
            phi_win  = MULTI_WIN
        else:
            phi_win = -1 * math.inf

        
        temp_phi_save_vic = self.victim_saved - self.prev_victim

        if temp_phi_save_vic == 0:
            phi_save_vic = 0
        else:
            phi_save_vic = temp_phi_save_vic * MULTI_SAVE

        temp_phi_ex_fire = self.fire_ex - self.prev_fire

        if temp_phi_ex_fire == 0:
            phi_ex_fire = 0
        else:
            phi_ex_fire = temp_phi_ex_fire * MULTI_EX
        



        


        rew = max(min(max(phi_dist_fire, phi_ex_fire), max(phi_dist_victim, phi_save_vic), phi_dist_med_fire, phi_dist_med_ff), phi_win)


        rew = rew/20

        print(f"dist fire: {phi_dist_fire}, EX fire: {phi_ex_fire}, dist victim: {phi_dist_victim}, save victim: {phi_save_vic}, dist med fire: {phi_dist_med_fire}, dist med FF: {phi_dist_med_ff}, win : {phi_win}, reward: {rew}")


        # update

        self.prev_victim = self.victim_saved



        return rew
    
    def can_move(self, agent_id, move):

        agent = self.agents[agent_id]
        
        if move == Moves.UP:
            x, y = int(agent.x - 1), int(agent.y)
        elif move == Moves.DOWN:
            x, y = int(agent.x + 1), int(agent.y)
        elif move == Moves.LEFT:
            x, y = int(agent.x), int(agent.y - 1)
        else:
            x, y = int(agent.x), int(agent.y + 1)

        return 0 <= x < self.n_grid and 0 <= y < self.n_grid

    def get_avail_agent_actions(self, agent_id):
        
        avail_actions = [0] * self.n_actions

        if (self.can_move(agent_id, Moves.UP)):
            avail_actions[0] = 1
        if (self.can_move(agent_id, Moves.DOWN)):
            avail_actions[1] = 1
        if (self.can_move(agent_id, Moves.LEFT)):
            avail_actions[2] = 1
        if (self.can_move(agent_id, Moves.RIGHT)):
            avail_actions[3] = 1

        return avail_actions
<<<<<<< HEAD
=======

>>>>>>> d1f5e922b4b43221a388944c90f234ae8836eab7
