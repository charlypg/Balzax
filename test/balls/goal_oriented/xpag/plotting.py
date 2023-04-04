import os
from matplotlib import figure
import numpy as np
from typing import List, Dict, Any, Union, Optional
from matplotlib import collections as mc

from xpag.agents.agent import Agent
from xpag.setters.setter import Setter
from xpag.tools.eval import SaveEpisode
from xpag.tools.timing import timing
from xpag.tools.logging import eval_log
from xpag.plotting.plotting import single_episode_plot
from xpag.tools.utils import DataType, datatype_convert, hstack, logical_or


def single_episode_plot_custom(
    filename: str,
    step_list: List[Dict[str, Any]],
    projection_function=lambda x: x[0:2],
    plot_env_function=None,
):
    """Plots an episode, using a 2D projection from observations, or
    from achieved and desired goals in the case of GoalEnv environments.
    """
    fig = figure.Figure()
    ax = fig.subplots(1)
    xmax = 1.0
    xmin = 0.0
    ymax = 1.0
    ymin = 0.0
    lines = []
    rgbs = []
    gx = []
    gy = []
    gax = []
    gay = []
    episode_length = len(step_list)
    goalenv = False
    for j, step in enumerate(step_list):
        if (
            isinstance(step["observation"], dict)
            and "achieved_goal" in step["observation"]
        ):
            goalenv = True
            x_obs = datatype_convert(
                step["observation"]["observation"][0], DataType.NUMPY
            )  # achieved_goal
            x_obs = x_obs.reshape((x_obs.shape[0] // 2), 2)
            x_obs_next = datatype_convert(
                step["next_observation"]["observation"][0], DataType.NUMPY
            )  # achieved_goal
            x_obs_next = x_obs_next.reshape((x_obs_next.shape[0] // 2), 2)
            gxy = datatype_convert(
                step["observation"]["desired_goal"][0], DataType.NUMPY
            )
            gxy = gxy.reshape((gxy.shape[0] // 2), 2)
            gx.append(gxy[:, 0])
            gy.append(gxy[:, 1])
            gaxy = datatype_convert(
                step["observation"]["achieved_goal"][0], DataType.NUMPY
            )
            gaxy = gaxy.reshape((gaxy.shape[0] // 2), 2)
            gax.append(gaxy[:, 0])
            gay.append(gaxy[:, 1])
        else:
            x_obs = projection_function(
                datatype_convert(step["observation"][0], DataType.NUMPY)
            )
            x_obs = x_obs.reshape((x_obs.shape[0] // 2), 2)
            x_obs_next = projection_function(
                datatype_convert(step["next_observation"][0], DataType.NUMPY)
            )
            x_obs_next = x_obs_next.reshape((x_obs_next.shape[0] // 2), 2)

        nb_goals = len(x_obs)
        green_rates = np.linspace(0.1, 0.9, nb_goals)

        if j == 0:
            for k, (o, no) in enumerate(zip(x_obs, x_obs_next)):
                lines.append([(o, no)])
                rgbs.append(
                    [
                        (
                            1.0 - j / episode_length / 2.0,
                            green_rates[nb_goals - 1 - k],
                            0.2 + j / episode_length / 2.0,
                            1,
                        )
                    ]
                )
        else:
            for k, (o, no) in enumerate(zip(x_obs, x_obs_next)):
                lines[k].append((o, no))
                rgbs[k].append(
                    (
                        1.0 - j / episode_length / 2.0,
                        green_rates[nb_goals - 1 - k],
                        0.2 + j / episode_length / 2.0,
                        1,
                    )
                )
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if plot_env_function is not None:
        plot_env_function(ax)
    if goalenv:
        ax.scatter(gx, gy, s=10, c="green", alpha=0.8)
        ax.scatter(gax, gay, s=10, c="red", alpha=0.8)
    for line, color in zip(lines, rgbs):
        ax.add_collection(mc.LineCollection(line, colors=color, linewidths=1.0))
    fig.savefig(filename, dpi=200)
    fig.clf()
    ax.cla()


def single_rollout_eval_custom(
    steps: int,
    eval_env: Any,
    env_info: Dict[str, Any],
    agent: Agent,
    setter: Setter,
    save_dir: Union[str, None] = None,
    plot_projection=None,
    save_episode: bool = False,
    env_datatype: Optional[DataType] = None,
    seed: Optional[int] = None,
):
    """Evaluation performed on a single run"""
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    interval_time, _ = timing()
    observation, _ = setter.reset(
        eval_env,
        *eval_env.reset(seed=master_rng.randint(1e9)),
        eval_mode=True,
    )
    if save_episode and save_dir is not None:
        save_ep = SaveEpisode(eval_env, env_info)
        save_ep.update()
    done = np.array(False)
    cumulated_reward = 0.0
    step_list = []
    while not done.max():
        obs = (
            observation
            if not env_info["is_goalenv"]
            else hstack(observation["observation"], observation["desired_goal"])
        )
        action = agent.select_action(obs, eval_mode=True)
        action_info = {}
        if isinstance(action, tuple):
            action_info = action[1]
            action = action[0]
        action = datatype_convert(action, env_datatype)
        next_observation, reward, terminated, truncated, info = setter.step(
            eval_env,
            observation,
            action,
            action_info,
            *eval_env.step(action),
            eval_mode=True,
        )
        done = logical_or(terminated, truncated)
        if save_episode and save_dir is not None:
            save_ep.update()
        cumulated_reward += reward.mean()
        step_list.append(
            {"observation": observation, "next_observation": next_observation}
        )
        observation = next_observation
    eval_log(
        steps,
        interval_time,
        cumulated_reward,
        None if not env_info["is_goalenv"] else info["is_success"].mean(),
        env_info,
        agent,
        save_dir,
    )
    if plot_projection is not None and save_dir is not None:
        os.makedirs(os.path.join(os.path.expanduser(save_dir), "plots"), exist_ok=True)
        single_episode_plot_custom(
            os.path.join(
                os.path.expanduser(save_dir),
                "plots",
                f"{steps:12}.png".replace(" ", "0"),
            ),
            step_list,
            projection_function=plot_projection,
            plot_env_function=None if not hasattr(eval_env, "plot") else eval_env.plot,
        )
    if save_episode and save_dir is not None:
        save_ep.save(0, os.path.expanduser(save_dir))
    timing()
