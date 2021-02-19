import json
from itertools import groupby
from operator import itemgetter
from collections import Counter

import numpy as np


def load(filename):
    with open(filename) as json_file_:
        data = json.load(json_file_)
    return data


def load_level_data(test, level, name="RL"):
    level_path = "static/levelData/"
    level_name = "-Test_%d_Level_%d.json" % (test, level)
    return load("%sOR%s" % (level_path, level_name)), load("%s%s%s" % (level_path, name, level_name))


def episode_per_agent(level_data):
    episode_ = np.array(level_data["environmentData"]["episode"])
    hist_per_agent = {}
    for agent in range(episode_.shape[1]):
        hist_per_agent[agent] = np.flip(episode_[:, agent, :2], -1).tolist()
    return hist_per_agent


def episode_per_timestep(level_data):
    episode = np.array(level_data["environmentData"]["episode"])[:, :, :2]
    episode_per_timestep = {}
    for timestep in range(episode.shape[0]):
        episode_per_timestep[timestep] = [{"x": int(agent[1]), "y": int(agent[0]), "id": index} for index, agent in
                                          enumerate(episode[timestep])]
    return episode_per_timestep


def agent_start_end(agent_episode):
    start = -1
    for i, (pos1, pos2) in enumerate(agent_episode):
        if pos1 != 0 and pos2 != 0 and start == -1:
            start = i
        elif pos1 == 0 and pos2 == 0 and start != -1:
            return start, i - 1
    return start, len(agent_episode) - 1


def num_agents(level_data):
    episode = np.array(level_data["environmentData"]["episode"])
    return episode.shape[1]


def agents(level_data):
    epi_agent = episode_per_agent(level_data)
    agents_ = []
    for agent in level_data["environmentData"]["agents"]:
        agent_start, agent_end = agent_start_end(epi_agent[agent["agent_index"]])
        agents_.append({  # "src": agent["initial_position"], "dest": agent["target"],
            "index": agent["agent_index"],
            "src_x": agent["initial_position"][0], "src_y": agent["initial_position"][1],
            "dest_x": agent["target"][0], "dest_y": agent["target"][1],
            "start_t": agent_start, "end_t": agent_end})
    keys = [("src", itemgetter("src_x", "src_y")),
            ("dest", itemgetter("dest_x", "dest_y")),
            ("src-dest", itemgetter("src_x", "src_y", "dest_x", "dest_y"))]
    for index_name, key in keys:
        for _, group in groupby(sorted(agents_, key=key), key=key):
            for i, agent in enumerate(sorted(group, key=itemgetter("start_t"))):
                agent["%s-index" % index_name] = i
    return sorted(agents_, key=lambda x: x["index"])


def get_groups(agents_):
    keys = [("src", itemgetter("src_x", "src_y")),
            ("dest", itemgetter("dest_x", "dest_y")),
            ("src-dest", itemgetter("src_x", "src_y", "dest_x", "dest_y"))]
    groups = {"src": [], "dest": [], "src-dest": []}
    for index_name, key in keys:
        for group_key, group in groupby(sorted(agents_, key=key), key=key):
            groups[index_name].append(group_key)
    return groups


def heatmap_data(level_data):
    agents_ = agents(level_data)
    epi_agent = episode_per_agent(level_data)
    grid = np.array(level_data["environmentData"]["grid"])
    heatmap = [[{} for j in range(grid.shape[0])] for i in range(grid.shape[1])]
    for agent_index, agent_episode in epi_agent.items():
        for (pos2, pos1) in agent_episode:
            values = [agents_[agent_index]["src_x"], agents_[agent_index]["src_y"],
                      agents_[agent_index]["dest_x"], agents_[agent_index]["dest_y"]]
            keys = ["src-%d-%d" % (values[0], values[1]),
                    "dest-%d-%d" % (values[2], values[3]),
                    "src-dest-%d-%d-%d-%d" % (values[0], values[1], values[2], values[3])]
            for key in keys:
                if key in heatmap[pos1][pos2]:
                    heatmap[pos1][pos2][key] += 1
                else:
                    heatmap[pos1][pos2][key] = 1
    return np.array(heatmap)


def heatmap_dif(heatmap1, heatmap2):
    heatmap1 = heatmap1.flatten()
    heatmap2 = heatmap2.flatten()
    result = []
    for index, (val1, val2) in enumerate(zip(heatmap1, heatmap2)):
        c = Counter(val1)
        c.subtract(val2)
        result.append({k: v for k, v in dict(c).items() if v != 0})
    return np.array(result)


def heatmap_combine(heatmap_data1, heatmap_data2):
    result = [[{} for _ in range(len(heatmap_data1[0]))] for _ in range(len(heatmap_data1))]
    for i, (row_lhs, row_rhs) in enumerate(zip(heatmap_data1, heatmap_data2)):
        for j, (cell_lhs, cell_rhs) in enumerate(zip(row_lhs, row_rhs)):
            result[i][j] = {"lhs": cell_lhs, "rhs": cell_rhs}
    return result


def distance_coordinates(level_data):
    foo = episode_per_agent(level_data)
    bar = {}
    for agent, hist in foo.items():
        coord = []
        distance = -1
        last_pos = [0, 0]
        for i, pos in enumerate(hist):
            if pos != last_pos:
                distance = distance + 1
            if pos != [0, 0]:
                coord.append([i, distance])
            last_pos = pos
        bar[agent] = coord
    return bar


def start_time(episode_data):
    return next(i for i, pos in enumerate(episode_data) if pos != [0, 0])


def finish_time(episode_data):
    if episode_data[-1] != [0, 0]:
        return -1
    return next(i for i, pos in reversed(list(enumerate(episode_data))) if pos != [0, 0]) + 1


def number_of_stops(episode_data):
    travel_episode = [pos for pos in episode_data if pos != [0, 0]]
    current_pos = [0, 0]
    stops = 0
    for pos in travel_episode:
        if current_pos == pos:
            stops += 1
        current_pos = pos
    return stops


def traveled_distance(episode_data):
    travel_episode = [pos for pos in episode_data if pos != [0, 0]]
    current_pos = [0, 0]
    distance = 0
    for pos in travel_episode:
        if current_pos != pos:
            distance += 1
        current_pos = pos
    return distance


def get_metrics(episode_per_agent_):
    result = {"n_finished": len([finish_time(v) for k, v in episode_per_agent_.items() if finish_time(v) != -1]),
              "total_time": max([finish_time(v) for k, v in episode_per_agent_.items()]),
              "mean_start_time": np.mean([start_time(v) for k, v in episode_per_agent_.items()]),
              "std_start_time": np.std([start_time(v) for k, v in episode_per_agent_.items()]),
              "mean_finish_time": np.mean([finish_time(v) for k, v in episode_per_agent_.items() if finish_time(v) != -1]),
              "std_finish_time": np.std([finish_time(v) for k, v in episode_per_agent_.items() if finish_time(v) != -1]),
              "mean_stops": np.mean([number_of_stops(v) for k, v in episode_per_agent_.items()]),
              "max_stops": max([number_of_stops(v) for k, v in episode_per_agent_.items()]),
              "mean_distance": np.mean([traveled_distance(v) for k, v in episode_per_agent_.items()])}
    return result


def analyse_episode(episode_data):
    steps = []
    travel_episode = [i for i, pos in enumerate(episode_data) if pos != [0, 0]]
    start = travel_episode[0]
    current_pos = [0, 0]
    moving = True
    for i in travel_episode:
        if current_pos == episode_data[i]:
            if moving:
                moving = False
                steps.append([start, i - 1])
        if current_pos != episode_data[i]:
            if not moving:
                start = i
                moving = True
        current_pos = episode_data[i]
    if moving:
        steps.append([start, travel_episode[-1]])
    return steps


def time_data_per_grp(level_data):
    episode_per_agent_ = episode_per_agent(level_data)
    agents_ = agents(level_data)
    groups = get_groups(agents_)
    time_data = {}
    for (x, y) in groups["src"]:
        agents_in_group = [agent["index"] for agent in agents_ if agent["src_x"] == x and agent["src_y"] == y]
        obj = {"agent_index": agents_in_group}
        for agent in agents_in_group:
            obj[str(agent)] = analyse_episode(episode_per_agent_[agent])
        time_data["src-%d-%d" % (x, y)] = obj
    return time_data
