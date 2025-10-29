import copy
import random
from collections import namedtuple
from operator import attrgetter

from flask import Flask, jsonify, request, send_from_directory, render_template
import numpy as np

# ===== Paths =====
ICON_DIR = "/Users/tartmsu/Desktop/wildfire-ai-model/img"  

api_file = 'api.txt'
try:
    with open(api_file, 'r') as file:
        for line in file:
            api_file = line.strip()
except FileNotFoundError:
    print(f"API FILE '{api_file}' WAS NOT FOUND.")

# ===== Map/Env config =====
CENTER_LAT = 34.232862882409094
CENTER_LNG = -118.0141139046774

# ~1.1km box; shrink if you want tighter view
DELTA = 0.01
SW_LAT, SW_LNG = CENTER_LAT - DELTA, CENTER_LNG - DELTA
NE_LAT, NE_LNG = CENTER_LAT + DELTA, CENTER_LNG + DELTA

N_GRID = 5
GMAPS_JS_API_KEY = api_file
TICK_SECONDS = 1.0

Agent = namedtuple("Agent", ["y", "x", "type_id"])

class WildFireEnv:
    def __init__(self, n_grid=N_GRID, method="hypRL", mode="train",
                 sw_lat=SW_LAT, sw_lng=SW_LNG, ne_lat=NE_LAT, ne_lng=NE_LNG):
        self.n_grid = n_grid
        self.method = method
        self.mode = mode

        # Entities (y, x)
        self.FF = [[2, 0], [3, 0]]    # Firefighter Helicopters (H)
        self.med = [[1, 0], [1, 0]]   # Medical Drones (D)
        self.fire = [[0, 1], [1, 2], [2, 1]]
        self.victims = [[0, 0], [1, 2]]

        self.original_FF = copy.deepcopy(self.FF)
        self.original_med = copy.deepcopy(self.med)
        self.original_fire = copy.deepcopy(self.fire)
        self.original_victims = copy.deepcopy(self.victims)

        self.FF_ID = 1000
        self.MED_ID = 2000
        self.FIRE_ID = 3000
        self.VICT_ID = 4000

        self.agents = {}
        self.n_agents = 0

        self.victim_saved = 0
        self.fire_ex = 0
        self.trunct = False
        self.max_step = 1000 if mode == "train" else 30000

        self.sw_lat, self.sw_lng = sw_lat, sw_lng
        self.ne_lat, self.ne_lng = ne_lat, ne_lng

        self.init_agents()

    def init_agents(self):
        ally_agents = []
        for ff in self.FF:
            ally_agents.append(Agent(ff[0], ff[1], self.FF_ID))
        for med in self.med:
            ally_agents.append(Agent(med[0], med[1], self.MED_ID))
        sorted_ally_agents = sorted(ally_agents, key=attrgetter("x", "y", "type_id"))
        self.agents = {i: sorted_ally_agents[i] for i in range(len(sorted_ally_agents))}
        self.n_agents = len(sorted_ally_agents)

    def transition(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        probs = {
            0: (0.9, 0.025, 0.025, 0.025, 0.025),  # up
            1: (0.025, 0.9, 0.025, 0.025, 0.025),  # down
            2: (0.025, 0.025, 0.9, 0.025, 0.025),  # left
            3: (0.025, 0.025, 0.025, 0.9, 0.025),  # right
            4: (0.025, 0.025, 0.025, 0.025, 0.9),  # stay
        }[action]
        dy, dx = random.choices(moves, weights=probs, k=1)[0]
        return dy, dx

    def step(self, actions=None):
        if actions is None:
            actions = [random.randint(0, 4) for _ in range(self.n_agents)]
        for agent_id, action in enumerate(actions):
            agent = self.agents[agent_id]
            dy, dx = self.transition(action)
            new_y = int(np.clip(agent.y + dy, 0, self.n_grid - 1))
            new_x = int(np.clip(agent.x + dx, 0, self.n_grid - 1))
            self.agents[agent_id] = Agent(new_y, new_x, agent.type_id)

        vistm_copy = copy.deepcopy(self.victims)
        fire_copy = copy.deepcopy(self.fire)
        for agent in self.agents.values():
            a = [agent.y, agent.x]
            if (a in fire_copy) and (agent.type_id == self.FF_ID):
                self.fire_ex += 1
                fire_copy.remove(a)
            if (a in vistm_copy) and (agent.type_id == self.MED_ID):
                self.victim_saved += 1
                vistm_copy.remove(a)
        self.victims, self.fire = vistm_copy, fire_copy

        terminated = (len(self.fire) == 0) and (len(self.victims) == 0)
        if terminated:
            self.trunct = True
        return terminated

    def reset(self):
        self.FF = copy.deepcopy(self.original_FF)
        self.med = copy.deepcopy(self.original_med)
        self.fire = copy.deepcopy(self.original_fire)
        self.victims = copy.deepcopy(self.original_victims)
        self.victim_saved = 0
        self.fire_ex = 0
        self.trunct = False
        self.init_agents()

    def cell_center_latlng(self, y, x):
        fy = (y + 0.5) / self.n_grid
        fx = (x + 0.5) / self.n_grid
        lat = self.ne_lat + fy * (self.sw_lat - self.ne_lat)  # down = smaller lat
        lng = self.sw_lng + fx * (self.ne_lng - self.sw_lng)  # right = larger lng
        return float(lat), float(lng)

    def state_json(self):
        center_lat = (self.ne_lat + self.sw_lat) / 2.0
        center_lng = (self.ne_lng + self.sw_lng) / 2.0

        agents = []
        for a in self.agents.values():
            lat, lng = self.cell_center_latlng(a.y, a.x)
            label = "H" if a.type_id == self.FF_ID else "D"
            agents.append({"y": a.y, "x": a.x, "lat": lat, "lng": lng, "label": label})

        fires = [{"y": fy, "x": fx, "lat": self.cell_center_latlng(fy, fx)[0],
                  "lng": self.cell_center_latlng(fy, fx)[1]} for (fy, fx) in self.fire]
        victims = [{"y": vy, "x": vx, "lat": self.cell_center_latlng(vy, vx)[0],
                    "lng": self.cell_center_latlng(vy, vx)[1]} for (vy, vx) in self.victims]

        return {
            "bbox": {"sw": {"lat": self.sw_lat, "lng": self.sw_lng},
                     "ne": {"lat": self.ne_lat, "lng": self.ne_lng}},
            "center": {"lat": center_lat, "lng": center_lng},
            "n_grid": self.n_grid,
            "agents": agents,
            "fires": fires,
            "victims": victims,
            "stats": {"fires_extinguished": self.fire_ex, "victims_saved": self.victim_saved},
        }

# ---- Flask app ----
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)
env = WildFireEnv(n_grid=N_GRID, sw_lat=SW_LAT, sw_lng=SW_LNG, ne_lat=NE_LAT, ne_lng=NE_LNG)

@app.route("/icons/<path:filename>")
def serve_icons(filename):
    allowed = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    if not filename.lower().endswith(allowed):
        return "Forbidden", 403
    return send_from_directory(ICON_DIR, filename)

@app.get("/")
def index():
    # render template and inject your Maps key + tick millis
    return render_template("index.html", key=GMAPS_JS_API_KEY, tick_ms=int(TICK_SECONDS*1000))

@app.get("/state")
def get_state():
    return jsonify(env.state_json())

@app.post("/tick")
def tick():
    actions = request.json.get("actions") if request.is_json else None
    env.step(actions)
    return jsonify(env.state_json())

@app.post("/reset")
def reset():
    env.reset()
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)
