from actor import DeticActor

from explorer.environ.utils import inspect_environ
from explorer_prelude import *


env_path = '/home/planner/dataset/MP3D/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'
scene_info = inspect_environ(env_path)
actors = [DeticActor(scene_info.nav_bbox, 5, (5, 5))]
envs = [Env(env_path)]
run_simulation(actors, envs)