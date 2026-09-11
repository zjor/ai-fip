from pathlib import Path

import mujoco.viewer


scene_path = Path(__file__).with_name("scene.xml").resolve()

mujoco.viewer.launch_from_path(str(scene_path))
