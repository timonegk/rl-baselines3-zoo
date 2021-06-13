"""
Plot training reward/success rate
"""
import argparse
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func
del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

# Activate seaborn
seaborn.set()

parser = argparse.ArgumentParser("Gather results, plot training reward/success")
parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
parser.add_argument("-e", "--env", help="Environment to include", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=14)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward"], type=str, default="reward")
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=10)

args = parser.parse_args()


algo = args.algo
env = args.env
log_path = os.path.join(args.exp_folder, algo)

x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.x_axis]
x_label = {"steps": "Timesteps (in Million)", "episodes": "Episodes", "time": "Walltime (in hours)"}[args.x_axis]

y_axis = {"success": "is_success", "reward": "r"}[args.y_axis]
y_label = {"success": "Training Success Rate", "reward": "Evaluation reward"}[args.y_axis]

dirs = [
    os.path.join(log_path, folder)
    for folder in os.listdir(log_path)
    if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
]

for folder in sorted(dirs):
    npz = np.load(os.path.join(folder, 'evaluations.npz'))
    timesteps = npz['timesteps']
    results = npz['results']
    if args.max_timesteps is not None:
        results = results[timesteps <= args.max_timesteps]
        timesteps = timesteps[timesteps <= args.max_timesteps]

    # Take mean for each of the evaluations
    results = np.mean(results, axis=1)

    # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
    if timesteps.shape[0] >= args.episode_window:
        name = folder.split('/')[-1]
        # Compute and plot rolling mean with window of size args.episode_window
        x, y_mean = window_func(timesteps, results, args.episode_window, np.mean)
        almost_there = np.where(y_mean >= 0.95*y_mean.max())[0][0]
        print(name, '– 5% Deviation of maximum is first reached at timestep', x[almost_there])
        plt.figure(y_label, figsize=args.figsize)
        plt.title(y_label, fontsize=args.fontsize)
        plt.xlabel(f"{x_label}", fontsize=args.fontsize)
        plt.ylabel(y_label, fontsize=args.fontsize)
        plt.ylim(0, 60)
        plt.plot(x / 1e6, y_mean, linewidth=2)
        plt.tight_layout()
        #plt.show()
        plt.savefig(name + '_eval')
        plt.close()
        #plt.show()

#plt.legend()
#plt.tight_layout()
#plt.show()
