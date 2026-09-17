"""SMP air-drop task MDP components.

Reuses the getup task's events (``reset_stand_counter``) and terminations
(``smp_too_low``, ``stood_up``, ``diverged``); the rewards here add the airborne
hold-pose phase on top of the shared ukemi/getup terms.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from smp.rl.tasks.getup_mini.mdp.events import *  # noqa: F401, F403
from smp.rl.tasks.getup_mini.mdp.terminations import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
