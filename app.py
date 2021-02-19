import json
import numpy as np
import flatland_data_processing as fdp
from flask import Flask, render_template
from transition_image import load_png_rotation

app = Flask(__name__)


@app.route('/')
def index():
    or_data, rl_data = fdp.load_level_data(4, 0, "RL")
    # get general data
    grid = np.vstack(or_data["environmentData"]["grid"])
    groups = fdp.get_groups(fdp.agents(or_data))
    n_agents = fdp.num_agents(or_data)
    # get specific data
    episode_per_timestep = {0: fdp.episode_per_timestep(or_data),
                            1: fdp.episode_per_timestep(rl_data)}
    episode_per_agent = {0: fdp.episode_per_agent(or_data),
                         1: fdp.episode_per_agent(rl_data)}
    heatmap_or = fdp.heatmap_data(or_data)
    heatmap_rl = fdp.heatmap_data(rl_data)
    heatmap_dif = fdp.heatmap_dif(heatmap_or, heatmap_rl)
    heatmap = np.vstack([heatmap_or.flatten(), heatmap_rl.flatten(), heatmap_dif]).swapaxes(0, 1).tolist()
    metrics = {0: fdp.get_metrics(episode_per_agent[0]),
               1: fdp.get_metrics(episode_per_agent[1])}
    time_data_or = fdp.time_data_per_grp(or_data)
    time_data_rl = fdp.time_data_per_grp(rl_data)
    time_data = {}
    for key in time_data_or.keys():
        time_data[key] = {0: time_data_or[key], 1: time_data_rl[key]}
    # set data for template
    data = {"images": load_png_rotation(),
            "grid": grid.flatten().tolist(),
            "nColumns": grid.shape[1],
            "groups": groups,
            "nAgents": n_agents,
            "episode": episode_per_timestep,
            "episode_per_agent": episode_per_agent,
            "heatmap": heatmap,
            "metrics": metrics,
            "time_data": time_data
            }
    return render_template("index.html", data=data)


if __name__ == '__main__':
    app.run(debug=True)
