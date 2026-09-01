"""StackPlanner 2.0 integration implemented as native slime plugins.

Production StackPlanner and ordinary slime training retain their existing
behavior unless the top-level launcher selects one of these hooks.
"""

from .config import Experiment, TrainingConfig, TrainingMode
from .model_factory import EpisodeModelFactory, make_openai_model_builder
from .routing import Backend, RoleRoute, RoleRouter
from .sessions import SessionKey, SessionManager

__all__ = [
    "Backend",
    "EpisodeModelFactory",
    "Experiment",
    "RoleRoute",
    "RoleRouter",
    "SessionKey",
    "SessionManager",
    "TrainingConfig",
    "TrainingMode",
    "make_openai_model_builder",
]
