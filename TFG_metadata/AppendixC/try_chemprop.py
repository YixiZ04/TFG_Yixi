import numpy as np
import matplotlib.pyplot as plt
from ray import tune
import optuna

N = 10000

search_space = {
  "depth": tune.qrandint(lower=2, upper=6, q=1),
  "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
  "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
  "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
}
ray_notebook = {k: np.array([v.sample() for _ in range(N)]) for k, v in search_space.items()}

cli_search_space = {
  "depth": tune.randint(lower=2, upper=6 + 1),  # `+ 1` because upper bound is exclusive
  "ffn_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
  "ffn_num_layers": tune.randint(lower=1, upper=2 + 1),
  "message_hidden_dim": tune.qrandint(lower=300, upper=2400, q=100),
}
ray_cli = {k: np.array([v.sample() for _ in range(N)]) for k, v in cli_search_space.items()}

study = optuna.create_study(direction = "minimize")

### Swapped "log=True" by "step=100"
optuna_search_space = {
  "mp_hidden_dim": lambda trial: trial.suggest_int("mp_hidden_dim", 300, 2400, step=100),
  "mp_depth": lambda trial: trial.suggest_int("mp_depth", 2, 6, step=1),
  "ffn_hidden_dim": lambda trial: trial.suggest_int("ffn_hidden_dim", 300, 2400, step=100),
  "ffn_layers": lambda trial: trial.suggest_int("ffn_layers", 1, 3, step=1),
}

optuna_notebook = {k: np.array([optuna_search_space[k](study.ask()) for _ in range(N)]) for k in optuna_search_space}

study_not_log = optuna.create_study(direction = "minimize")
optuna_not_log = {
  "mp_hidden_dim": lambda trial: trial.suggest_int("mp_hidden_dim", 300, 2400, log=False),
  "mp_depth": lambda trial: trial.suggest_int("mp_depth", 2, 6, log=False),
  "ffn_hidden_dim": lambda trial: trial.suggest_int("ffn_hidden_dim", 300, 2400, log=False),
  "ffn_layers": lambda trial: trial.suggest_int("ffn_layers", 1, 3, log=False),
}

optuna_not_log = {k: np.array([optuna_not_log[k](study_not_log.ask()) for _ in range(N)]) for k in optuna_not_log}

optuna_notebook = {
  "depth": optuna_notebook["mp_depth"],
  "ffn_hidden_dim": optuna_notebook["ffn_hidden_dim"],
  "ffn_num_layers": optuna_notebook["ffn_layers"],
  "message_hidden_dim": optuna_notebook["mp_hidden_dim"],
}

optuna_not_log = {
  "depth": optuna_not_log["mp_depth"],
  "ffn_hidden_dim": optuna_not_log["ffn_hidden_dim"],
  "ffn_num_layers": optuna_not_log["ffn_layers"],
  "message_hidden_dim": optuna_not_log["mp_hidden_dim"],
}

fig, axes = plt.subplots(4, 4, figsize=(16, 12))

sources = [
  (ray_notebook, "Ray Notebook", "steelblue"),
  (ray_cli, "Ray CLI", "cornflowerblue"),
  (optuna_notebook, "Optuna Notebook", "darkorange"),
  (optuna_not_log, "Optuna Not Log", "goldenrod"),
]

param_keys = list(ray_notebook.keys())

for j, key in enumerate(param_keys):
  for i, (data, label, color) in enumerate(sources):
      ax = axes[i, j]
      vals = data[key]
      lo, hi = vals.min(), vals.max()
      bins = np.arange(lo - 0.5, hi + 1.5, 1)
      ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.85)
      ax.set_xlim(lo - 1, hi + 1)
      if i == 0:
          ax.set_title(key, fontsize=11, fontweight="bold")
      if j == 0:
          ax.set_ylabel(label, fontsize=11)

plt.tight_layout()
plt.savefig("4x4_comparison.png", dpi=150, bbox_inches="tight")
plt.show()