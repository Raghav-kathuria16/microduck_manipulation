import sys, os, math
from pathlib import Path
import numpy as np
import mujoco
import imageio

ws = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ws / "scripts"))
from test_spoon_lemon_balance import create_spoon_lemon_scene_xml, create_policy_with_steadicam

sim_duration_sec = 25.0  # 25.0 seconds at authentic 1x natural physics time (>= 20s)
target_vx = 0.20
fps = 25

mp4_path = str(ws / "docs" / "media" / "lemon_spoon_balance.mp4")
webp_path = str(ws / "docs" / "media" / "lemon_spoon_balance.webp")
policy_path = str(ws / "policies" / "onnx" / "payload_clamping_locomotion.onnx")

print(f"[1X LEMON] Initializing simulation with proven smooth policy {policy_path}...")
xml_str = create_spoon_lemon_scene_xml(0.50)
model = mujoco.MjModel.from_xml_string(xml_str)
model.opt.timestep = 0.002
data = mujoco.MjData(model)

policy, is_rl = create_policy_with_steadicam(model, data, policy_path)
policy.head_pos_steadicam = 0.0

spoon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
lemon_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lemon")
beak_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "beak")
beak_dof = model.jnt_dofadr[beak_jid]
beak_qpos = model.jnt_qposadr[beak_jid]
jaw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
spoon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spoon_free")
spoon_qadr = model.jnt_qposadr[spoon_jid]
lemon_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lemon_free")
lemon_qadr = model.jnt_qposadr[lemon_jid]

mujoco.mj_resetData(model, data)
data.qpos[0:3] = [0.0, 0.0, 0.125]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
for i, idx in enumerate(policy.joint_qpos_indices):
    data.qpos[idx] = policy.default_pose[i]
data.qpos[beak_qpos] = 1.550
data.qfrc_applied[beak_dof] = -0.45
mujoco.mj_forward(model, data)

p_jaw = data.xpos[jaw_bid]
R_jaw = data.xmat[jaw_bid].reshape(3, 3)
data.qpos[spoon_qadr:spoon_qadr+3] = p_jaw + R_jaw @ np.array([-0.0119, 0.0, -0.0465])
q_jaw = np.empty(4); mujoco.mju_mat2Quat(q_jaw, data.xmat[jaw_bid])
q_s = np.empty(4); mujoco.mju_mulQuat(q_s, q_jaw, np.array([0.0, 0.7071068, 0.7071068, 0.0]))
data.qpos[spoon_qadr+3:spoon_qadr+7] = q_s
mujoco.mj_forward(model, data)

p_spoon = data.xpos[spoon_bid]
R_spoon = data.xmat[spoon_bid].reshape(3, 3)
data.qpos[lemon_qadr:lemon_qadr+3] = p_spoon + R_spoon @ np.array([0.0, 0.0052, 0.070])
data.qpos[lemon_qadr+3:lemon_qadr+7] = q_s
mujoco.mj_forward(model, data)

# Settle 150 steps
for _ in range(150):
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qvel[0:6] = 0.0
    data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * (1.550 - data.qpos[beak_qpos]) - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
    mujoco.mj_step(model, data)

policy.set_vel_cmd(target_vx, 0.0, 0.0)

# Pre-step 200 physics steps so robot is actively walking in frame 0 (NO static pause)
for step in range(200):
    data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * (1.550 - data.qpos[beak_qpos]) - 0.3 * data.qvel[beak_dof], -1.0, 0.2))
    if step % 10 == 0:
        act = policy.infer()
        R_trunk = data.xmat[1].reshape(3, 3)
        h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
        h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))
        rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
        policy.head_pos_steadicam = np.clip(policy.head_pos_steadicam + rate_cmd * 0.02, -0.40, 0.40)
        act[5] = 0.0 + policy.head_pos_steadicam
        act[6] = 0.0 - policy.head_pos_steadicam
        act[7] = 0.0
        act[8] = 0.0
        policy.apply_action(act)
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, width=1280, height=720)
vopt = mujoco.MjvOption()
vopt.geomgroup[3] = 0
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = 1
cam.distance = 0.72
cam.elevation = -14.0
cam.azimuth = 138.0

total_sim_steps = int(sim_duration_sec / model.opt.timestep)
sim_steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))

# Write directly to mp4_path at 1x real-time natural physics speed (25 fps for 25.0s = 625 frames)
writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)
frame_count = 0

print(f"[1X LEMON] Recording {sim_duration_sec:.1f}s at authentic 1X natural speed ({total_sim_steps} steps, {fps} fps)...", flush=True)
for step in range(total_sim_steps):
    t_sim = step * model.opt.timestep
    data.qfrc_applied[beak_dof] = float(np.clip(-0.45 + 10.0 * (1.550 - data.qpos[beak_qpos]) - 0.3 * data.qvel[beak_dof], -1.0, 0.2))

    if step % 10 == 0:
        act = policy.infer()
        R_trunk = data.xmat[1].reshape(3, 3)
        h_fwd = np.array([R_trunk[0, 0], R_trunk[1, 0], 0.0])
        h_fwd /= (np.linalg.norm(h_fwd) + 1e-6)
        R_s = data.xmat[spoon_bid].reshape(3, 3)
        b_norm = R_s @ np.array([0.0, 1.0, 0.0])
        spoon_pitch = math.asin(np.clip(float(np.dot(b_norm, h_fwd)), -0.999, 0.999))
        rate_cmd = np.clip(1.20 * spoon_pitch, -0.30, 0.30)
        policy.head_pos_steadicam = np.clip(policy.head_pos_steadicam + rate_cmd * 0.02, -0.40, 0.40)
        act[5] = 0.0 + policy.head_pos_steadicam
        act[6] = 0.0 - policy.head_pos_steadicam
        act[7] = 0.0
        act[8] = 0.0
        policy.apply_action(act)

    mujoco.mj_step(model, data)

    if step % sim_steps_per_frame == 0:
        renderer.update_scene(data, camera=cam, scene_option=vopt)
        writer.append_data(renderer.render())
        frame_count += 1
        if frame_count % (fps * 5) == 0:
            p_s = data.xpos[spoon_bid]
            R_s = data.xmat[spoon_bid].reshape(3, 3)
            b_c = p_s + R_s @ np.array([0.0, 0.0052, 0.070])
            dev = np.linalg.norm(data.xpos[lemon_bid] - b_c) * 1000.0
            print(f"  Frame {frame_count:4d}/{int(sim_duration_sec*fps)} (t={t_sim:5.1f}s): walked {data.qpos[0]:+5.2f}m | lemon dev={dev:4.1f}mm", flush=True)

writer.close()
print(f"[1X LEMON SUCCESS] Saved 1x MP4 ({mp4_path}) - {frame_count} frames, duration = {frame_count/fps:.2f}s", flush=True)

# Generate high-resolution 600px wide WebP animation at 1x natural playback speed
cmd_webp = f'ffmpeg -y -i "{mp4_path}" -vf "fps=10,scale=600:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 4 -q:v 38 -loop 0 "{webp_path}"'
os.system(cmd_webp)

root_mp4 = str(ws / "docs" / "media" / "lemon_spoon_balance.mp4")
root_webp = str(ws / "docs" / "media" / "lemon_spoon_balance.webp")
os.system(f'cp "{mp4_path}" "{root_mp4}"')
os.system(f'cp "{webp_path}" "{root_webp}"')

sz = os.path.getsize(webp_path) / 1e6
print(f"[1X LEMON SUCCESS] Saved 1x WebP ({webp_path}) - Size: {sz:.2f} MB", flush=True)
