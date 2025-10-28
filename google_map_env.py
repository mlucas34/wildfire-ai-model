
import copy
import math
import random
from collections import namedtuple
from operator import attrgetter

from flask import Flask, jsonify, request, Response, send_from_directory
import numpy as np
import os
import time 

# point this to your real folder with the PNGs
ICON_DIR = "/Users/tartmsu/Desktop/wildfire-ai-model/img"

# ====== CONFIG ======
center_lat = 34.232862882409094
center_lng = -118.0141139046774

SW_LAT = center_lat - 0.01
SW_LNG = center_lng - 0.01
NE_LAT = center_lat + 0.01
NE_LNG = center_lng + 0.01

N_GRID = 5
GMAPS_JS_API_KEY = "KEY"  # <-- REQUIRED
TICK_SECONDS = 1.5  # how often the page will request a step
# ====================

Agent = namedtuple("Agent", ["y", "x", "type_id"])

class WildFireEnv:
    def __init__(self, n_grid=N_GRID, method="hypRL", mode="train",
                 sw_lat=SW_LAT, sw_lng=SW_LNG, ne_lat=NE_LAT, ne_lng=NE_LNG):
        self.n_grid = n_grid
        self.method = method
        self.mode = mode
        self.s = 0

        # Entities (y, x)
        self.FF = [[2, 0], [3, 0]]   # Helicopters (H)
        self.med = [[1, 0], [1, 0]]  # Drones (D)
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
        self.trajectory = []
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
        sorted_ally_agents = sorted(
            ally_agents, key=attrgetter("x", "y", "type_id"), reverse=False
        )
        self.agents = {i: sorted_ally_agents[i] for i in range(len(sorted_ally_agents))}
        self.n_agents = len(sorted_ally_agents)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def transition(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        transition_probabilities = {
            0: (0.9, 0.025, 0.025, 0.025, 0.025),
            1: (0.025, 0.9, 0.025, 0.025, 0.025),
            2: (0.025, 0.025, 0.9, 0.025, 0.025),
            3: (0.025, 0.025, 0.025, 0.9, 0.025),
            4: (0.025, 0.025, 0.025, 0.025, 0.9),
        }
        dy, dx = random.choices(moves, weights=transition_probabilities[action], k=1)[0]
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
            a_coords = [agent.y, agent.x]
            if (a_coords in fire_copy) and (agent.type_id == self.FF_ID):
                self.fire_ex += 1
                fire_copy.remove(a_coords)
            if (a_coords in vistm_copy) and (agent.type_id == self.MED_ID):
                self.victim_saved += 1
                vistm_copy.remove(a_coords)
        self.victims = vistm_copy
        self.fire = fire_copy

        terminated = (len(self.fire) == 0) and (len(self.victims) == 0)
        if terminated:
            self.trunct = True

        self.s +=1
        return terminated

    def reset(self):
        self.FF = copy.deepcopy(self.original_FF)
        self.med = copy.deepcopy(self.original_med)
        self.fire = copy.deepcopy(self.original_fire)
        self.victims = copy.deepcopy(self.original_victims)
        self.victim_saved = 0
        self.fire_ex = 0
        self.s = 0
        self.trunct = False
        self.trajectory = []
        self.init_agents()

    def cell_center_latlng(self, y, x):
        fy = (y + 0.5) / self.n_grid
        fx = (x + 0.5) / self.n_grid
        lat = self.ne_lat + fy * (self.sw_lat - self.ne_lat)
        lng = self.sw_lng + fx * (self.ne_lng - self.sw_lng)
        return float(lat), float(lng)

    def state_json(self):
        center_lat = (self.ne_lat + self.sw_lat) / 2.0
        center_lng = (self.ne_lng + self.sw_lng) / 2.0

        agents = []
        for a in self.agents.values():
            lat, lng = self.cell_center_latlng(a.y, a.x)
            label = "H" if a.type_id == self.FF_ID else "D"
            agents.append({
                "y": a.y, "x": a.x, "type_id": a.type_id,
                "lat": lat, "lng": lng, "label": label
            })

        fires = []
        for (fy, fx) in self.fire:
            lat, lng = self.cell_center_latlng(fy, fx)
            fires.append({"y": fy, "x": fx, "lat": lat, "lng": lng})

        victims = []
        for (vy, vx) in self.victims:
            lat, lng = self.cell_center_latlng(vy, vx)
            victims.append({"y": vy, "x": vx, "lat": lat, "lng": lng})

        return {
            "bbox": {"sw": {"lat": self.sw_lat, "lng": self.sw_lng},
                     "ne": {"lat": self.ne_lat, "lng": self.ne_lng}},
            "center": {"lat": center_lat, "lng": center_lng},
            "n_grid": self.n_grid,
            "agents": agents,
            "fires": fires,
            "victims": victims,
            "stats": {"fires_extinguished": self.fire_ex, "victims_saved": self.victim_saved}
        }

# ============== Flask App ==============
app = Flask(__name__)
env = WildFireEnv(n_grid=N_GRID, sw_lat=SW_LAT, sw_lng=SW_LNG, ne_lat=NE_LAT, ne_lng=NE_LNG)

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Live Gridworld on Google Maps</title>
    <style>
      html, body, #map { height: 100%; margin:0; padding:0; }
      .hud {
        position:absolute; z-index:999; background:#ffffffcc; padding:8px 12px; border-radius:8px;
        left:12px; top:60px; font-family:sans-serif; font-size:14px;
      }
      .legend span { display:inline-block; margin-right:10px; }
      .dot { width:10px; height:10px; border-radius:50%; display:inline-block; vertical-align:middle; margin-right:4px; }
      .controls {
        position:absolute; z-index:999; right:12px; top:12px; background:#ffffffcc; padding:8px 12px; border-radius:8px;
        font-family:sans-serif; font-size:14px;
      }
      select { padding:4px; }
    </style>
    <script>
      let map;
      let gridLines = [];
      let agentMarkers = [];
      let fireMarkers = [];
      let victimMarkers = [];
      let mapType = 'hybrid';

      async function fetchState() {
        const res = await fetch('/state');
        return await res.json();
      }

      async function tick() {
        await fetch('/tick', {method: 'POST'});
        const state = await fetchState();
        updateHUD(state);
        ensureGrid(state);
        updateMarkers(state);
      }

      function ensureGrid(state) {
        if (!map.__grid_init) {
          drawGrid(map, state.bbox, state.n_grid);
          map.__grid_init = true;
        }
      }

      function drawGrid(map, bbox, n) {
        const sw = bbox.sw, ne = bbox.ne;
        for (const l of gridLines) l.setMap(null);
        gridLines = [];
        // horizontal
        for (let r=0; r<=n; r++) {
          const lat = ne.lat + (r/n)*(sw.lat - ne.lat);
          const path = [{lat: lat, lng: sw.lng}, {lat: lat, lng: ne.lng}];
          gridLines.push(new google.maps.Polyline({
            path, map, strokeOpacity: 0.6, strokeWeight: 1
          }));
        }
        // vertical
        for (let c=0; c<=n; c++) {
          const lng = sw.lng + (c/n)*(ne.lng - sw.lng);
          const path = [{lat: ne.lat, lng}, {lat: sw.lat, lng}];
          gridLines.push(new google.maps.Polyline({
            path, map, strokeOpacity: 0.6, strokeWeight: 1
          }));
        }
      }

      function clearMarkers(arr) {
        for (const m of arr) m.setMap(null);
        arr.length = 0;
      }

      function updateMarkers(state) {
        clearMarkers(agentMarkers);
        clearMarkers(fireMarkers);
        clearMarkers(victimMarkers);

        // Agents -> PNG icons from /icons folder
        for (const a of state.agents) {
          const iconUrl = (a.label === 'H')
            ? '/icons/FF.png'
            : '/icons/medic.png';
          const m = new google.maps.Marker({
            position: {lat: a.lat, lng: a.lng},
            map,
            icon: {
              url: iconUrl,
              scaledSize: new google.maps.Size(40, 40),
              anchor: new google.maps.Point(16, 16)
            },
            title: (a.label === 'H') ? 'Helicopter' : 'Drone'
          });
          agentMarkers.push(m);
        }

        // Fires
        for (const f of state.fires) {
          const m = new google.maps.Marker({
            position: {lat: f.lat, lng: f.lng},
            map,
            icon: {
              url: '/icons/fire.png',
              scaledSize: new google.maps.Size(40, 40),
              anchor: new google.maps.Point(14, 14)
            },
            title: 'Fire'
          });
          fireMarkers.push(m);
        }

        // Victims
        for (const v of state.victims) {
          const m = new google.maps.Marker({
            position: {lat: v.lat, lng: v.lng},
            map,
            icon: {
              url: '/icons/victim.png',
              scaledSize: new google.maps.Size(30, 30),
              anchor: new google.maps.Point(14, 14)
            },
            title: 'Victim'
          });
          victimMarkers.push(m);
        }
      }  // <-- MISSING in your version; closes updateMarkers!

      function updateHUD(state) {
        const stats = state.stats || {fires_extinguished:0, victims_saved:0};
        const el = document.getElementById('hud');
        el.innerHTML = `
          <div><b>Live Gridworld</b></div>
          <div>Grid: ${state.n_grid}×${state.n_grid}</div>
          <div>Fires extinguished: ${stats.fires_extinguished}</div>
          <div>Victims saved: ${stats.victims_saved}</div>
        `;
      }

      function onMapTypeChange(sel) {
        mapType = sel.value;
        map.setMapTypeId(mapType);
      }

      async function initMap() {
        const state = await fetchState();
        map = new google.maps.Map(document.getElementById('map'), {
          center: state.center,
          zoom: 15,
          mapTypeId: mapType,
          streetViewControl: false,
          fullscreenControl: false
        });
        updateHUD(state);
        ensureGrid(state);
        updateMarkers(state);
        setInterval(tick, """ + str(int(TICK_SECONDS * 1000)) + """);
      }
    </script>
    <script async defer
      src="https://maps.googleapis.com/maps/api/js?key=""" + GMAPS_JS_API_KEY + """&callback=initMap&v=weekly"></script>
  </head>
  <body>
    <div id="map"></div>
    <div id="hud" class="hud">Loading…</div>
    <div class="controls">
      <div><b>Map Type</b></div>
      <select onchange="onMapTypeChange(this)">
        <option value="hybrid" selected>Hybrid (Satellite + Labels)</option>
        <option value="satellite">Satellite</option>
        <option value="terrain">Terrain</option>
        <option value="roadmap">Roadmap</option>
      </select>
    </div>
  </body>
</html>
"""

@app.route("/icons/<path:filename>")
def serve_icons(filename):
    allowed = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    if not filename.lower().endswith(allowed):
        return "Forbidden", 403
    return send_from_directory(ICON_DIR, filename)

@app.get("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")

@app.get("/state")
def get_state():
    return jsonify(env.state_json())

@app.post("/tick")
def tick():
    if request.is_json and "actions" in request.json:
        actions = request.json["actions"]
    else:
        actions = None
    env.step(actions)
    return jsonify(env.state_json())

@app.post("/reset")
def reset():
    env.reset()
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=True)
