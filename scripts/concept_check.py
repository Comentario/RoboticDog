"""Load the concept model, set the standing pose and optionally render it (--render). Standing dynamics are validated in phase 2."""
import sys, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("robot/concept.xml")
d = mujoco.MjData(m)
print(f"total mass: {m.body_subtreemass[1]:.2f} kg | actuators: {m.nu}")

# standing pose: roll=0, thigh=+0.8, knee=-1.6 (feet under the hips)
stand = np.tile([0.0, 0.8, -1.6], 4)
d.ctrl[:] = stand
for j, v in zip(range(7, 7 + 12), stand):
    d.qpos[j] = v
mujoco.mj_forward(m, d)

if "--render" in sys.argv:
    r = mujoco.Renderer(m, 800, 1280)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.12]; cam.distance = 0.9
    cam.azimuth, cam.elevation = 140, -20
    r.update_scene(d, cam)
    from PIL import Image
    Image.fromarray(r.render()).save("docs/02_render_concepto.png"); print("render saved")
    sys.exit()

