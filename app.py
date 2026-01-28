from flask import *
import pandas as pd
import osmium as osm
import numpy as np
import matplotlib.pyplot as plt
import heapq
import sys
from scipy.spatial import KDTree
import io
import base64
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('Agg')

sheets = ['employees', 'vehicles', 'baseline', 'metadata']
osmfile = 'GeoData/bengaluru.osm'

nodes = {}
xnodes = {}
r_nodes = {}
edges = {}
r_edges = {}
ways = {}
ls_lat = []
ls_lng = []

def load_excel(filename):
    testcase = dict()
    try:
        for sheet in sheets:
            testcase[sheet] = pd.read_excel(filename, sheet_name=sheet)
        print(f"Input {filename} Successful.")
        return testcase
    except Exception as e:
        print(f"Error: {e}")
        return None

R = 6731
def haversine_distance(nd1, nd2):
    dlat = np.radians(abs(nodes[nd1][0] - nodes[nd2][0]))
    dlon = np.radians(abs(nodes[nd1][1] - nodes[nd2][1]))
    d = 2 * R * np.arcsin(np.sqrt((np.sin(dlat / 2)**2 + np.cos(np.radians(nodes[nd1][0])) * np.cos(np.radians(nodes[nd2][0])) * np.sin(dlon / 2)**2)))
    return d

class OSMHandler(osm.SimpleHandler):
    def node(self, n):
        nodes[n.id] = (n.location.lat, n.location.lon)
        xnodes[(n.location.lat, n.location.lon)] = n.id
        edges[n.id] = []
        r_edges[n.id] = []

    def way(self, w):
        ways[w.id] = w.tags
        nds = w.nodes

        road = False
        oneway = False
        access = True
        for t in w.tags:
            if t.k == 'highway':
                road = True
            if t.k == 'oneway' and t.v == 'yes':
                oneway = True
                break
            if t.k == 'access' and (t.k == 'no' or t.k == 'private'):
                access = False
            if t.k == 'motor_vehicle' and t.v == 'no':
                access = False
        if road and access:
            for n in nds:
                r_nodes[n.ref] = nodes[n.ref]
                ls_lat.append(nodes[n.ref][0])
                ls_lng.append(nodes[n.ref][1])
            ls_lat.append(None)
            ls_lng.append(None)
            
            nc = len(nds)
            for i in range(1, nc):
                d = haversine_distance(nds[i - 1].ref, nds[i].ref)
                edges[nds[i - 1].ref].append((nds[i].ref, d))
                if not oneway:
                    edges[nds[i].ref].append((nds[i - 1].ref, d))
                r_edges[nds[i].ref].append((nds[i - 1].ref, d))
                if not oneway:
                    r_edges[nds[i - 1].ref].append((nds[i].ref, d))

handler = OSMHandler()
handler.apply_file(osmfile)

points = np.array(list(r_nodes.values()))
tree = KDTree(points)

print('Pre Computation Complete!')

def plot_route(route_lat, route_lng):
    fig = Figure()
    ax = fig.subplots()
    ax.plot(ls_lat, ls_lng, 'b')
    ax.plot(route_lat, route_lng, 'r')
    ax.set_xlim([12.8, 13.2])
    ax.set_ylim([77.5, 77.9])
    ax.set_title('Bengaluru')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

def nearest_node(loc):
    md, p = tree.query(loc, k=1)
    p = tuple(points[p])
    nrst = xnodes[p]
    #print(f'Node {nrst} found {md * 10**3} m away')
    return nrst

def dijkstras(dest):
    pq = []
    dist = {}
    for n in r_nodes.keys():
        dist[n] = sys.maxsize
    dist[dest] = 0
    heapq.heappush(pq, (0, dest))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for v, w in r_edges[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def reconstruct_path(came_from, cur):
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    return path[::-1]

def astar(src, dest):
    open_set = []
    heapq.heappush(open_set, (0, src))

    came_from = {} # path reconstruction
    g_dist = {src: 0}
    f_dist = {src: haversine_distance(src, dest)}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == dest:
            return reconstruct_path(came_from, cur), g_dist[cur]
        for ngh, dist in edges[cur]:
            tg = g_dist[cur] + dist
            if ngh not in g_dist or tg < g_dist[ngh]:
                came_from[ngh] = cur
                g_dist[ngh] = tg
                f_dist[ngh] = tg + haversine_distance(ngh, dest)
                heapq.heappush(open_set, (f_dist[ngh], ngh))
    return None, float('inf')

def optimal_route(src, dest):
    route, length = astar(src, dest)
    #print('heuristic_route:', route)
    #print('Length:', length)
    route_lat = []
    route_lng = []
    if route is not None:
        for n in route:
            route_lat.append(r_nodes[n][0])
            route_lng.append(r_nodes[n][1])
        route_lat.append(None)
        route_lng.append(None)
    return route, length, plot_route(route_lat, route_lng)

def solve(file):
    answers = []
    tc = load_excel(file)
    employees, vehicles, baseline, metadata = tc['employees'], tc['vehicles'], tc['baseline'], tc['metadata']

    tc_id = metadata.iat[0, 1]
    print(f'ID: {tc_id}')

    drop = nearest_node((employees.iat[0, 4], employees.iat[0, 5]))
    #dist = dijkstras(drop)

    for emp in employees.itertuples():
        pick = nearest_node((emp.pickup_lat, emp.pickup_lng))

        #print(f'{emp.employee_id}:')
        #print('Optimal Path Length:', dist[pick])
        answers.append(optimal_route(pick, drop))
    return answers

#APP
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/output', methods = ['GET', 'POST'])
def output():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and file.filename.endswith(('.xls', '.xlsx')):
            res = solve(file)
            return render_template('output.html', result = res)
        else:
            return "Invalid file format. Please upload an Excel file.", 400

    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug = True)