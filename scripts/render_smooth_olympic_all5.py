import sys, os, math
from pathlib import Path
import numpy as np
import mujoco
import imageio
import onnxruntime as ort

ws = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ws / "scripts"))
from test_duck_racer_lemon_balance import create_racer_scene_xml, DUCK_CONFIGS, JOINT_NAMES
from infer_policy import DEFAULT_POSE

sim_duration_sec = 80.0  # 80.0s simulation -> 40.0s video at 2x
fps = 25  # 25 fps -> 2000 frames
raw_path = "/tmp/smooth_olympic_raw_all5_80s.mp4"
mp4_path = str(ws / "docs" / "media" / "duck_racer_competition_2x.mp4")
webp_path = str(ws / "docs" / "media" / "duck_racer_competition_2x.webp")
policy_path = str(ws / "policies" / "onnx" / "payload_clamping_locomotion.onnx")

print(f"[OLYMPIC ALL-5 ACTIVE] Generating track scene XML with smooth policy {policy_path}...")
xml = create_racer_scene_xml()
xml = xml.replace('pos="2.0 0.0 0.0005" size="6.0 0.88 0.0005"', 'pos="10.0 0.0 0.0005" size="14.0 0.88 0.0005"')
for div in ["lane_div_0", "lane_div_1", "lane_div_2", "lane_div_3", "lane_div_4", "lane_div_5"]:
    xml = xml.replace(f'name="{div}" type="box" pos="2.0', f'name="{div}" type="box" pos="10.0')
xml = xml.replace('size="6.0 0.012 0.001"', 'size="14.0 0.012 0.001"')
xml = xml.replace('name="barrier_left"  type="box" pos="2.0', 'name="barrier_left"  type="box" pos="10.0')
xml = xml.replace('name="barrier_right" type="box" pos="2.0', 'name="barrier_right" type="box" pos="10.0')
xml = xml.replace('size="6.0 0.025 0.04"', 'size="14.0 0.025 0.04"')

model = mujoco.MjModel.from_xml_string(xml)
model.opt.timestep = 0.002
data = mujoco.MjData(model)

session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

class ActiveOlympicDuck:
    def __init__(self, duck_id, cfg, target_vx):
        self.duck_id = duck_id
        self.cfg = cfg
        self.target_vx = target_vx
        self.prefix = f"d{duck_id}_"
        self.trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}trunk_base")
        self.jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}jaw_soft")
        self.spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}spoon")
        self.lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}lemon")
        self.trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}trunk_base_freejoint")
        self.trunk_qadr = model.jnt_qposadr[self.trunk_jid]
        self.trunk_dof = model.jnt_dofadr[self.trunk_jid]
        self.beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}beak")
        self.beak_qpos = model.jnt_qposadr[self.beak_jid]
        self.beak_dof = model.jnt_dofadr[self.beak_jid]
        self.spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}spoon_free")
        self.spoon_qadr = model.jnt_qposadr[self.spoon_jid]
        self.spoon_dof = model.jnt_dofadr[self.spoon_jid]
        self.lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}lemon_free")
        self.lemon_qadr = model.jnt_qposadr[self.lemon_jid]
        self.lemon_dof = model.jnt_dofadr[self.lemon_jid]
        self.sensor_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{self.prefix}imu_ang_vel")
        self.sensor_gyro_adr = model.sensor_adr[self.sensor_gyro_id]
        self.act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}{name}") for name in JOINT_NAMES]
        self.beak_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}beak")
        self.joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in self.act_ids]
        self.joint_dof_indices = [int(model.jnt_dofadr[model.actuator_trnid[a, 0]]) for a in self.act_ids]
        self.default_pose = DEFAULT_POSE[:14].copy()
        self.default_pose[5] = 0.3491
        self.default_pose[6] = 0.3491
        self.last_action = np.zeros(14, dtype=np.float32)
        self.current_vx = 0.0
        self.head_pos_steadicam = 0.0
        self.target_clamp_angle = 1.550
        self.clamp_torque = 0.45
        self.q_s = np.array([0.0, 0.7071068, 0.7071068, 0.0])

    def reset_posture(self):
        y0 = self.cfg["y"]
        data.qpos[self.trunk_qadr:self.trunk_qadr+3] = [0.0, y0, 0.125]
        data.qpos[self.trunk_qadr+3:self.trunk_qadr+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.trunk_dof:self.trunk_dof+6] = 0.0
        for i, idx in enumerate(self.joint_qpos_indices):
            data.qpos[idx] = self.default_pose[i]
            data.qvel[self.joint_dof_indices[i]] = 0.0
        for i, act_id in enumerate(self.act_ids):
            data.ctrl[act_id] = self.default_pose[i]
        data.qpos[self.beak_qpos] = self.target_clamp_angle
        data.qvel[self.beak_dof] = 0.0
        data.qfrc_applied[self.beak_dof] = -self.clamp_torque
        data.ctrl[self.beak_act_id] = 1.0

    def place_spoon(self):
        p_jaw = data.xpos[self.jaw_bid]
        R_jaw = data.xmat[self.jaw_bid].reshape(3, 3)
        data.qpos[self.spoon_qadr:self.spoon_qadr+3] = p_jaw + R_jaw @ np.array([-0.0119, 0.0, -0.0465])
        q_jaw = np.empty(4); mujoco.mju_mat2Quat(q_jaw, data.xmat[self.jaw_bid])
        q_s = np.empty(4); mujoco.mju_mulQuat(q_s, q_jaw, np.array([0.0, 0.7071068, 0.7071068, 0.0]))
        data.qpos[self.spoon_qadr+3:self.spoon_qadr+7] = q_s
        data.qvel[self.spoon_dof:self.spoon_dof+6] = 0.0
        self.q_s = q_s

    def place_lemon(self):
        p_spoon = data.xpos[self.spoon_bid]
        R_spoon = data.xmat[self.spoon_bid].reshape(3, 3)
        data.qpos[self.lemon_qadr:self.lemon_qadr+3] = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
        data.qpos[self.lemon_qadr+3:self.lemon_qadr+7] = self.q_s
        data.qvel[self.lemon_dof:self.lemon_dof+6] = 0.0

    def step_clamp(self):
        beak_err = self.target_clamp_angle - data.qpos[self.beak_qpos]
        data.qfrc_applied[self.beak_dof] = float(np.clip(-self.clamp_torque + 10.0 * beak_err - 0.3 * data.qvel[self.beak_dof], -1.0, 0.2))

    def step_control(self):
        if self.current_vx < self.target_vx:
            self.current_vx = min(self.target_vx, self.current_vx + 0.008)
        base_ang_vel = data.sensordata[self.sensor_gyro_adr:self.sensor_gyro_adr+3].astype(np.float32)
        R_trunk = data.xmat[self.trunk_bid].reshape(3, 3)
        proj_grav = (R_trunk.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
        qpos_rel = np.array([data.qpos[idx] - self.default_pose[i] for i, idx in enumerate(self.joint_qpos_indices)], dtype=np.float32)
        qvel = np.array([data.qvel[idx] for idx in self.joint_dof_indices], dtype=np.float32)
        last_act = self.last_action.copy()
        yaw = math.atan2(R_trunk[1, 0], R_trunk[0, 0])
        y_err = float(data.xpos[self.trunk_bid][1] - self.cfg["y"])
        target_yaw = math.atan2(-y_err, 0.60)
        yaw_err = (yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
        cmd_wz = float(np.clip(-2.5 * yaw_err, -0.45, 0.45))
        cmd = np.zeros(13, dtype=np.float32)
        cmd[0] = self.current_vx
        cmd[2] = cmd_wz

        obs = np.concatenate([base_ang_vel, proj_grav, qpos_rel, qvel, last_act, cmd]).astype(np.float32)[np.newaxis, :]
        action = session.run(None, {session.get_inputs()[0].name: obs})[0][0].astype(np.float32)
        self.last_action = action.copy()

        h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
        h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)
        R_s = data.xmat[self.spoon_bid].reshape(3, 3)
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))
        rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
        self.head_pos_steadicam = np.clip(self.head_pos_steadicam + rate_cmd * 0.02, -0.40, 0.40)

        action[5] = 0.0 + self.head_pos_steadicam
        action[6] = 0.0 - self.head_pos_steadicam
        action[7] = 0.0
        action[8] = 0.0

        targets = self.default_pose + action
        for i, act_id in enumerate(self.act_ids):
            data.ctrl[act_id] = targets[i]

# All paces >= 0.185 m/s so NO duck stalls:
# Ruby Coral: 0.240 m/s (2nd place)
# Riviera Sky: 0.185 m/s (5th place)
# Duck Gold: 0.220 m/s (3rd place)
# Emerald Green: 0.260 m/s (1st place Winner)
# Royal Amethyst: 0.200 m/s (4th place)
pacing = [0.240, 0.185, 0.220, 0.260, 0.200]
ducks = [ActiveOlympicDuck(c["id"], c, pacing[i]) for i, c in enumerate(DUCK_CONFIGS)]

for d in ducks: d.reset_posture()
mujoco.mj_forward(model, data)
for d in ducks: d.place_spoon()
mujoco.mj_forward(model, data)
for d in ducks: d.place_lemon()
mujoco.mj_forward(model, data)

for _ in range(150):
    for d in ducks:
        data.qpos[d.trunk_qadr:d.trunk_qadr+3] = [0.0, d.cfg["y"], 0.125]
        data.qvel[d.trunk_dof:d.trunk_dof+6] = 0.0
        d.step_clamp()
    mujoco.mj_step(model, data)

# Warmstart sprint steps so ducks are actively moving from frame 0 (NO static pause)
for step in range(200):
    for d in ducks: d.step_clamp()
    if step % 10 == 0:
        for d in ducks: d.step_control()
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, width=1280, height=720)
vopt = mujoco.MjvOption()
vopt.geomgroup[3] = 0

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 2.45
cam.elevation = -22.0
cam.azimuth = 68.0

total_sim_steps = int(sim_duration_sec / model.opt.timestep)
sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
decimation = 10
sim_dt = model.opt.timestep

print(f"[OLYMPIC ALL-5 ACTIVE] Recording {sim_duration_sec:.1f}s race ({total_sim_steps} steps, {fps} fps)...", flush=True)
writer = imageio.get_writer(raw_path, fps=fps, codec="libx264", quality=8)
frame_count = 0
smooth_lookat_x = float(np.mean([data.qpos[d.trunk_qadr] for d in ducks])) + 0.45

for step in range(total_sim_steps):
    t_sim = step * sim_dt
    for d in ducks:
        d.step_clamp()

    if step % decimation == 0:
        for d in ducks:
            d.step_control()

    mujoco.mj_step(model, data)

    if step % sim_steps_per_frame == 0:
        xs = [float(data.qpos[d.trunk_qadr]) for d in ducks]
        pack_mean_x = float(np.mean(xs))
        target_x = pack_mean_x + 0.45
        smooth_lookat_x = 0.96 * smooth_lookat_x + 0.04 * target_x
        cam.lookat = [smooth_lookat_x, 0.0, 0.16]

        renderer.update_scene(data, camera=cam, scene_option=vopt)
        writer.append_data(renderer.render())
        frame_count += 1

        if frame_count % (fps * 5) == 0:
            xs_str = ", ".join([f"D{d.duck_id}:{data.qpos[d.trunk_qadr]:.2f}m" for d in ducks])
            print(f"  Frame {frame_count:4d}/{int(sim_duration_sec*fps)} (t={t_sim:5.1f}s) | Progress: [{xs_str}]", flush=True)

writer.close()
print(f"[OLYMPIC ALL-5 ACTIVE] Saved {frame_count} raw frames to {raw_path}", flush=True)

cmd_2x = f'ffmpeg -y -i "{raw_path}" -filter:v "setpts=0.5*PTS" -r 30 "{mp4_path}"'
os.system(cmd_2x)
print(f"[OLYMPIC ALL-5 ACTIVE] Saved 2x video to {mp4_path}", flush=True)

root_mp4 = str(ws / "docs" / "media" / "duck_racer_competition_2x.mp4")
root_webp = str(ws / "docs" / "media" / "duck_racer_competition_2x.webp")
os.system(f'cp "{mp4_path}" "{root_mp4}"')

# Generate optimized WebP animation (< 3MB for instant GitHub Camo rendering)
cmd_webp = f'ffmpeg -y -i "{mp4_path}" -vf "fps=8,scale=420:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 50 -loop 0 "{webp_path}"'
os.system(cmd_webp)
os.system(f'cp "{webp_path}" "{root_webp}"')

sz = os.path.getsize(webp_path) / 1e6
print(f"[OLYMPIC ALL-5 ACTIVE SUCCESS] Saved WebP to {webp_path} - Size: {sz:.2f} MB", flush=True)

if os.path.exists(raw_path):
    os.remove(raw_path)
