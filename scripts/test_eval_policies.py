#!/usr/bin/env python3
"""Evaluate ground pick policies with visual servoing and BEST_alpha_stand."""
import os
import sys
import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIP_DIR = os.path.dirname(_SCRIPT_DIR)
_EXTERNAL_REPO = os.path.join(_MANIP_DIR, ".external_repo")
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "scripts"))
sys.path.insert(0, os.path.join(_EXTERNAL_REPO, "src"))
_RL_DIR = os.path.join(os.path.dirname(_MANIP_DIR), "microduck_rl")
sys.path.insert(0, os.path.join(_RL_DIR, "scripts"))
sys.path.insert(0, _MANIP_DIR)

from infer_policy import (
    load_bam_model,
    load_mujoco_with_bam,
    PolicyInference,
    BAM_KP_FW,
    BAM_VIN_MIN,
)
from vision.barbell_pose_estimator import BarbellPoseEstimator

PHYSICAL_CLAMP_XML = os.path.join(
    _EXTERNAL_REPO, "src/mjlab_microduck/robot/microduck/scene_loadcarry_physical_clamp.xml"
)
STAND_POLICY = os.path.join(_MANIP_DIR, "policies/onnx/BEST_alpha_stand.onnx")

def eval_policy(gp_policy_path, name="Policy"):
    print(f"\n=======================================================")
    print(f" Testing {name}: {os.path.basename(gp_policy_path)}")
    print(f"=======================================================")
    
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, names = load_mujoco_with_bam(PHYSICAL_CLAMP_XML, bam_model, 0.005, 0.0, BAM_VIN_MIN)
    
    cyl_bid = model.body("cylinder").id
    cyl_jid = model.joint("cylinder_free").id
    beak_jid = model.joint("beak").id
    mouth_sid = model.site("mouth_tip").id
    cyl_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_geom")
    jaw_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_soft_lower_collision")
    mouth_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soft_mouth_top_collision")
    beak_qadr = model.jnt_qposadr[beak_jid]
    
    model.body_mass[cyl_bid] = 0.100  # 100g
    
    policy = PolicyInference(
        model, data,
        standing_onnx_path=STAND_POLICY,
        ground_pick_onnx_path=gp_policy_path,
        ground_pick_period=4.0,
        use_projected_gravity=True,
        new_cmd_obs=True,
        bam_ctrl=bam_ctrl,
    )
    
    estimator = BarbellPoseEstimator(img_width=640, img_height=480, fovy_deg=45.0)
    
    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.125]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.qpos[beak_qadr] = 1.500
    
    c_adr = model.jnt_qposadr[cyl_jid]
    data.qpos[c_adr:c_adr + 3] = [0.115, 0.0, 0.0282]
    data.qpos[c_adr + 3:c_adr + 7] = [0.7071068, 0.7071068, 0.0, 0.0]
    
    bam_ctrl.reset(data.qpos)
    mujoco.mj_forward(model, data)
    
    for step in range(80):
        if step % 4 == 0:
            act = policy.infer()
            policy.apply_action(act)
        bam_ctrl.q_target[policy.beak_act_id] = 1.500
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        
    policy.trigger_ground_pick()
    
    initial_z = data.xpos[cyl_bid, 2]
    max_z = initial_z
    con_count = 0
    top_con_count = 0
    bot_con_count = 0
    
    for step in range(1000):
        if policy.ground_pick_mode:
            policy.update_ground_pick_phase(0.005)
            
        phi = policy.ground_pick_phase if policy.ground_pick_mode else 1.0
        
        if step % 4 == 0:
            est = estimator.estimate_from_sim(model, data, add_noise=False)
            nudge = estimator.compute_nudge_command(est)
            
            policy.head_offset = np.array([
                est.pos_mouth[0], est.pos_mouth[1], est.pos_mouth[2], est.yaw_rel
            ], dtype=np.float32)
            policy.body_cmd = np.array([
                nudge["cmd_vy"], nudge["cmd_wz"],
                np.hypot(est.pos_mouth[0], est.pos_mouth[1]),
                1.0 if nudge["is_aligned"] else 0.0,
                0.0, 0.0
            ], dtype=np.float32)
            policy._update_command()
            
            if policy.ground_pick_mode:
                policy.command[0] = np.cos(2 * np.pi * policy.ground_pick_phase)
                policy.command[1] = np.sin(2 * np.pi * policy.ground_pick_phase)
                policy.command[2] = 0.0
                
            act = policy.infer()
            
            # Visual servoing during approach / under-scoop
            if policy.ground_pick_mode and 0.15 <= phi < 0.28:
                dy = data.site_xpos[mouth_sid, 1] - data.xpos[cyl_bid, 1]
                dz = data.site_xpos[mouth_sid, 2] - (data.xpos[cyl_bid, 2] - 0.006)
                act[7] += float(np.clip(-2.5 * dy, -0.4, 0.4))
                if dz > 0:
                    act[5] -= float(np.clip(4.0 * dz, 0.0, 0.45))
                    act[6] -= float(np.clip(2.0 * dz, 0.0, 0.25))
                    
            # Neck elevation during stand recovery
            if phi >= 0.65 or not policy.ground_pick_mode:
                act[5] = 0.3491 - policy.default_pose[5]
                act[6] = 0.1745 - policy.default_pose[6]
                
            policy.apply_action(act)
            
        # 15th Actuator Control
        if policy.ground_pick_mode:
            if phi < 0.24:
                bam_ctrl.q_target[policy.beak_act_id] = 1.950
            else:
                bam_ctrl.q_target[policy.beak_act_id] = 1.620
        else:
            bam_ctrl.q_target[policy.beak_act_id] = 1.620
            
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        
        top_con = 0
        bot_con = 0
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if cyl_geom_id in (g1, g2):
                if mouth_geom_id in (g1, g2):
                    top_con += 1
                if jaw_geom_id in (g1, g2):
                    bot_con += 1
        if top_con > 0: top_con_count += 1
        if bot_con > 0: bot_con_count += 1
        if top_con > 0 or bot_con > 0: con_count += 1
        
        c_z = data.xpos[cyl_bid, 2]
        if c_z > max_z:
            max_z = c_z
            
        if step % 50 == 0:
            trunk_z = data.xpos[1, 2] * 1000.0
            m_z = data.site_xpos[mouth_sid, 2] * 1000.0
            q_beak = data.qpos[beak_qadr]
            print(f"step {step:4d} | phi {phi:5.2f} | trunk {trunk_z:5.1f} | cyl {c_z*1000:5.1f} | mouth {m_z:5.1f} | beak {q_beak:.3f} | top {top_con} bot {bot_con}")
            
    print(f"\nResult: Initial Z: {initial_z*1000:.1f} mm, Max Z: {max_z*1000:.1f} mm, Lift: {(max_z-initial_z)*1000:.1f} mm")
    print(f"Contacts: Top {top_con_count} steps, Bot {bot_con_count} steps, Total contact: {con_count} steps")
    print(f"Final trunk Z: {data.xpos[1, 2]*1000:.1f} mm")

if __name__ == "__main__":
    p_headup = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_headup.onnx")
    p_phys = os.path.join(_MANIP_DIR, "policies/onnx/loadcarry_ground_pick_physical.onnx")
    if os.path.exists(p_headup):
        eval_policy(p_headup, "loadcarry_ground_pick_headup")
    if os.path.exists(p_phys):
        eval_policy(p_phys, "loadcarry_ground_pick_physical")
