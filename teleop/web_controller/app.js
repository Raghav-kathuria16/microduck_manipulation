// MicroDuck V3 Multi-Condition Interactive Controller Logic

const state = {
  // Current raw targets
  targetVx: 0.25,
  targetVy: 0.0,
  targetWz: 0.0,
  
  // Rate-limited output (sent to robot)
  vx: 0.25,
  vy: 0.0,
  wz: 0.0,

  // Payload weight settings
  payloadMass: 300.0, // grams
  bareRobotMass: 737.2, // grams
  resetRequested: false,

  // Velocity envelope limits
  maxVxFwd: 0.25,
  maxVxBwd: 0.10,
  maxVy: 0.10,
  maxWz: 0.60,
  accelVx: 1.2,   // m/s^2
  accelWz: 1.5,   // rad/s^2 (cantilever inertia protection)

  keysDown: new Set(),
  joystickActive: false,
  connected: false,
  lastSendTime: performance.now(),
};

// DOM Elements - Weight
const sliderPayloadMass = document.getElementById('sliderPayloadMass');
const sliderMassVal = document.getElementById('sliderMassVal');
const dispWeightMass = document.getElementById('dispWeightMass');
const dispTotalMass = document.getElementById('dispTotalMass');
const dispWeightRatio = document.getElementById('dispWeightRatio');
const headerPayloadBadge = document.getElementById('headerPayloadBadge');
const wpresetBtns = document.querySelectorAll('.wpreset-btn');
const btnResetPose = document.getElementById('btnResetPose');

// DOM Elements - Velocity & Limits
const sliderMaxVx = document.getElementById('sliderMaxVx');
const sliderMaxWz = document.getElementById('sliderMaxWz');
const sliderAccelWz = document.getElementById('sliderAccelWz');
const dispMaxVx = document.getElementById('dispMaxVx');
const dispMaxWz = document.getElementById('dispMaxWz');
const dispAccelWz = document.getElementById('dispAccelWz');
const vpresetBtns = document.querySelectorAll('.vpreset-btn');

// DOM Elements - Joystick & Controls
const canvas = document.getElementById('joystickCanvas');
const ctx = canvas.getContext('2d');
const connectionStatus = document.getElementById('connectionStatus');

// DOM Elements - Telemetry Readouts
const simStatusTag = document.getElementById('simStatusTag');
const telemNeck = document.getElementById('telemNeck');
const badgeNeck = document.getElementById('badgeNeck');
const meterNeck = document.getElementById('meterNeck');
const boxNeck = document.getElementById('boxNeck');

const telemPitch = document.getElementById('telemPitch');
const badgePitch = document.getElementById('badgePitch');
const meterPitch = document.getElementById('meterPitch');
const boxPitch = document.getElementById('boxPitch');

const telemSpeed = document.getElementById('telemSpeed');
const meterSpeed = document.getElementById('meterSpeed');
const targetSpeedSub = document.getElementById('targetSpeedSub');

const telemZ = document.getElementById('telemZ');
const telemRoll = document.getElementById('telemRoll');
const meterZ = document.getElementById('meterZ');
const telemStepSub = document.getElementById('telemStepSub');

// Canvas dimensions
const center = { x: canvas.width / 2, y: canvas.height / 2 };
const radius = 85;
let knobPos = { x: center.x, y: center.y };

// --- WEIGHT CONTROLLER LOGIC ---
function updateWeightUI(massG) {
  state.payloadMass = massG;
  const totalG = state.bareRobotMass + massG;
  const ratioPct = (massG / state.bareRobotMass) * 100.0;

  dispWeightMass.innerHTML = `${massG.toFixed(0)} <small>g</small>`;
  dispTotalMass.innerHTML = `${totalG.toFixed(1)} <small>g</small>`;
  dispWeightRatio.textContent = `${ratioPct.toFixed(1)}%`;
  sliderMassVal.textContent = `${massG.toFixed(0)} g`;
  sliderPayloadMass.value = massG;
  headerPayloadBadge.textContent = `Payload: ${massG.toFixed(0)}g (${ratioPct.toFixed(1)}%)`;

  // Highlight active preset button
  wpresetBtns.forEach((btn) => {
    const btnMass = parseFloat(btn.getAttribute('data-mass'));
    if (Math.abs(btnMass - massG) < 1.0) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

sliderPayloadMass.addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  updateWeightUI(val);
});

wpresetBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const val = parseFloat(btn.getAttribute('data-mass'));
    updateWeightUI(val);
  });
});

// Reset Pose Handler
btnResetPose.addEventListener('click', () => {
  state.resetRequested = true;
  btnResetPose.style.transform = 'scale(0.95)';
  setTimeout(() => { btnResetPose.style.transform = ''; }, 150);
});

// --- VELOCITY PRESET LOGIC ---
function updateVelocityPreset(vxVal) {
  state.maxVxFwd = vxVal;
  sliderMaxVx.value = vxVal;
  dispMaxVx.textContent = `${vxVal.toFixed(2)} m/s`;

  if (state.keysDown.size === 0 && !state.joystickActive) {
    state.targetVx = vxVal;
  }

  vpresetBtns.forEach((btn) => {
    const bVx = parseFloat(btn.getAttribute('data-vx'));
    if (Math.abs(bVx - vxVal) < 0.005) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

vpresetBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const val = parseFloat(btn.getAttribute('data-vx'));
    updateVelocityPreset(val);
  });
});

sliderMaxVx.addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  updateVelocityPreset(val);
});

sliderMaxWz.addEventListener('input', (e) => {
  state.maxWz = parseFloat(e.target.value);
  dispMaxWz.textContent = `${state.maxWz.toFixed(2)} rad/s`;
  updateFromKeyboard();
});

sliderAccelWz.addEventListener('input', (e) => {
  state.accelWz = parseFloat(e.target.value);
  dispAccelWz.textContent = `${state.accelWz.toFixed(2)} rad/s²`;
});

// --- VIRTUAL JOYSTICK ---
function drawJoystick() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Outer Ring
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
  ctx.lineWidth = 3;
  ctx.stroke();

  // Directional Guides
  ctx.beginPath();
  ctx.moveTo(center.x, center.y - radius);
  ctx.lineTo(center.x, center.y + radius);
  ctx.moveTo(center.x - radius, center.y);
  ctx.lineTo(center.x + radius, center.y);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Deadzone circle
  ctx.beginPath();
  ctx.arc(center.x, center.y, 14, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.fill();

  // Connecting Line
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(knobPos.x, knobPos.y);
  ctx.strokeStyle = state.joystickActive ? 'rgba(56, 189, 248, 0.6)' : 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Knob
  ctx.beginPath();
  ctx.arc(knobPos.x, knobPos.y, 26, 0, Math.PI * 2);
  const knobGrad = ctx.createRadialGradient(knobPos.x, knobPos.y, 2, knobPos.x, knobPos.y, 26);
  knobGrad.addColorStop(0, state.joystickActive ? '#38bdf8' : '#64748b');
  knobGrad.addColorStop(1, state.joystickActive ? '#0284c7' : '#334155');
  ctx.fillStyle = knobGrad;
  ctx.shadowColor = state.joystickActive ? '#38bdf8' : 'transparent';
  ctx.shadowBlur = state.joystickActive ? 16 : 0;
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function handlePointer(px, py) {
  const rect = canvas.getBoundingClientRect();
  const x = px - rect.left;
  const y = py - rect.top;

  const dx = x - center.x;
  const dy = y - center.y;
  const dist = Math.hypot(dx, dy);

  if (dist > radius) {
    knobPos.x = center.x + (dx / dist) * radius;
    knobPos.y = center.y + (dy / dist) * radius;
  } else {
    knobPos.x = x;
    knobPos.y = y;
  }

  const normY = -(knobPos.y - center.y) / radius;
  const normX = (knobPos.x - center.x) / radius;

  const deadzone = 0.08;
  const clampedY = Math.abs(normY) < deadzone ? 0 : normY;
  const clampedX = Math.abs(normX) < deadzone ? 0 : normX;

  if (clampedY >= 0) {
    state.targetVx = clampedY * state.maxVxFwd;
  } else {
    state.targetVx = clampedY * state.maxVxBwd;
  }
  state.targetWz = -clampedX * state.maxWz;

  drawJoystick();
}

function resetJoystick() {
  state.joystickActive = false;
  knobPos = { x: center.x, y: center.y };
  if (state.keysDown.size === 0) {
    // Return to selected preset nominal speed rather than halting abruptly
    state.targetVx = state.maxVxFwd;
    state.targetWz = 0.0;
  }
  drawJoystick();
}

canvas.addEventListener('mousedown', (e) => {
  state.joystickActive = true;
  handlePointer(e.clientX, e.clientY);
});

window.addEventListener('mousemove', (e) => {
  if (state.joystickActive) {
    handlePointer(e.clientX, e.clientY);
  }
});

window.addEventListener('mouseup', resetJoystick);

canvas.addEventListener('touchstart', (e) => {
  state.joystickActive = true;
  const touch = e.touches[0];
  handlePointer(touch.clientX, touch.clientY);
  e.preventDefault();
});

window.addEventListener('touchmove', (e) => {
  if (state.joystickActive && e.touches.length > 0) {
    const touch = e.touches[0];
    handlePointer(touch.clientX, touch.clientY);
  }
});

window.addEventListener('touchend', resetJoystick);

// --- KEYBOARD TELEOP ---
function updateFromKeyboard() {
  if (state.joystickActive) return;

  const fwd = state.keysDown.has('w') || state.keysDown.has('arrowup');
  const bwd = state.keysDown.has('s') || state.keysDown.has('arrowdown');
  const left = state.keysDown.has('a') || state.keysDown.has('arrowleft');
  const right = state.keysDown.has('d') || state.keysDown.has('arrowright');
  const strafeL = state.keysDown.has('q');
  const strafeR = state.keysDown.has('e');

  if (fwd && !bwd) {
    state.targetVx = state.maxVxFwd;
  } else if (bwd && !fwd) {
    state.targetVx = -state.maxVxBwd;
  } else {
    state.targetVx = state.maxVxFwd; // Default to cruise at current envelope
  }

  if (left && !right) {
    state.targetWz = state.maxWz;
  } else if (right && !left) {
    state.targetWz = -state.maxWz;
  } else {
    state.targetWz = 0.0;
  }

  if (strafeL && !strafeR) {
    state.targetVy = state.maxVy;
  } else if (strafeR && !strafeL) {
    state.targetVy = -state.maxVy;
  } else {
    state.targetVy = 0.0;
  }

  const normY = state.targetVx > 0 ? (state.targetVx / Math.max(0.01, state.maxVxFwd)) : (state.targetVx / Math.max(0.01, state.maxVxBwd));
  const normX = -state.targetWz / Math.max(0.01, state.maxWz);
  knobPos.x = center.x + normX * radius * 0.8;
  knobPos.y = center.y - normY * radius * 0.8;
  drawJoystick();
}

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) {
    if (k === ' ') {
      emergencyStop();
      return;
    }
    state.keysDown.add(k);
    highlightButton(k, true);
    updateFromKeyboard();
  }
});

window.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  state.keysDown.delete(k);
  highlightButton(k, false);
  updateFromKeyboard();
});

function highlightButton(key, active) {
  const map = {
    w: 'btnW', arrowup: 'btnW',
    s: 'btnS', arrowdown: 'btnS',
    a: 'btnA', arrowleft: 'btnA',
    d: 'btnD', arrowright: 'btnD',
    q: 'btnQ',
    e: 'btnE',
  };
  const id = map[key];
  if (id) {
    const el = document.getElementById(id);
    if (el) {
      if (active) el.classList.add('active');
      else el.classList.remove('active');
    }
  }
}

document.querySelectorAll('.dpad-btn, .strafe-btn').forEach((btn) => {
  const key = btn.getAttribute('data-key');
  if (!key) return;

  btn.addEventListener('mousedown', () => {
    state.keysDown.add(key);
    btn.classList.add('active');
    updateFromKeyboard();
  });

  btn.addEventListener('mouseup', () => {
    state.keysDown.delete(key);
    btn.classList.remove('active');
    updateFromKeyboard();
  });

  btn.addEventListener('mouseleave', () => {
    state.keysDown.delete(key);
    btn.classList.remove('active');
    updateFromKeyboard();
  });
});

function emergencyStop() {
  state.keysDown.clear();
  document.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
  state.targetVx = 0.0;
  state.targetVy = 0.0;
  state.targetWz = 0.0;
  state.vx = 0.0;
  state.vy = 0.0;
  state.wz = 0.0;
  resetJoystick();
}

document.getElementById('btnStop').addEventListener('click', emergencyStop);

// --- 50 HZ CONTROL DISPATCH LOOP ---
function controlLoop() {
  const now = performance.now();
  const dt = Math.max(0.001, (now - state.lastSendTime) / 1000.0);
  state.lastSendTime = now;

  // Rate-limit vx
  const dvxMax = state.accelVx * dt;
  const dvx = state.targetVx - state.vx;
  state.vx += Math.max(-dvxMax, Math.min(dvxMax, dvx));

  // Rate-limit vy
  const dvyMax = 1.0 * dt;
  const dvy = state.targetVy - state.vy;
  state.vy += Math.max(-dvyMax, Math.min(dvyMax, dvy));

  // Rate-limit wz
  const dwzMax = state.accelWz * dt;
  const dwz = state.targetWz - state.wz;
  state.wz += Math.max(-dwzMax, Math.min(dwzMax, dwz));

  // POST Command to Localhost Server
  sendCommandToServer();
}

async function sendCommandToServer() {
  const payload = {
    vx: parseFloat(state.vx.toFixed(4)),
    vy: parseFloat(state.vy.toFixed(4)),
    wz: parseFloat(state.wz.toFixed(4)),
    target_vx: state.targetVx,
    target_wz: state.targetWz,
    payload_mass: state.payloadMass,
    reset_requested: state.resetRequested,
  };

  // Consume one-shot reset request
  state.resetRequested = false;

  try {
    const res = await fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      if (!state.connected) {
        state.connected = true;
        connectionStatus.className = 'status-indicator live';
      }
    }
  } catch (err) {
    if (state.connected) {
      state.connected = false;
      connectionStatus.className = 'status-indicator disconnected';
    }
  }
}

// --- 10 HZ LIVE TELEMETRY RECEIVER LOOP ---
async function pollTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    if (res.ok) {
      const data = await res.json();
      updateTelemetryUI(data);
    }
  } catch (err) {
    // Non-fatal
  }
}

function updateTelemetryUI(data) {
  // 1. Neck Angle
  if (data.neck !== undefined) {
    const neckVal = parseFloat(data.neck);
    telemNeck.innerHTML = `${neckVal >= 0 ? '+' : ''}${neckVal.toFixed(1)}<small>°</small>`;
    
    // Meter (0 to 30 deg mapped 20% to 100%)
    const pctNeck = Math.max(10, Math.min(100, 50 + (neckVal / 25.0) * 50));
    meterNeck.style.width = `${pctNeck}%`;

    if (neckVal > 5.0) {
      badgeNeck.textContent = 'UPRIGHT';
      badgeNeck.style.color = '#34d399';
      boxNeck.className = 'gauge-card highlight-box-green';
    } else if (neckVal >= -2.0) {
      badgeNeck.textContent = 'NEUTRAL';
      badgeNeck.style.color = '#fbbf24';
      boxNeck.className = 'gauge-card';
    } else {
      badgeNeck.textContent = 'DROOPING';
      badgeNeck.style.color = '#f87171';
      boxNeck.className = 'gauge-card';
    }
  }

  // 2. Torso Pitch
  if (data.pitch !== undefined) {
    const pitchVal = parseFloat(data.pitch);
    telemPitch.innerHTML = `${pitchVal >= 0 ? '+' : ''}${pitchVal.toFixed(1)}<small>°</small>`;
    
    const pctPitch = Math.max(10, Math.min(100, 50 - (pitchVal / 10.0) * 50));
    meterPitch.style.width = `${pctPitch}%`;

    if (pitchVal < -0.5) {
      badgePitch.textContent = 'LEAN BACK';
      badgePitch.style.color = '#38bdf8';
    } else if (pitchVal > 2.0) {
      badgePitch.textContent = 'LEAN FWD';
      badgePitch.style.color = '#f87171';
    } else {
      badgePitch.textContent = 'LEVEL';
      badgePitch.style.color = '#94a3b8';
    }
  }

  // 3. Velocity
  if (data.speed !== undefined) {
    const spd = parseFloat(data.speed);
    telemSpeed.innerHTML = `${spd >= 0 ? '+' : ''}${spd.toFixed(2)}<small>m/s</small>`;
    const pctSpd = Math.max(5, Math.min(100, (spd / 0.45) * 100));
    meterSpeed.style.width = `${pctSpd}%`;
    targetSpeedSub.textContent = `Target: ${state.vx.toFixed(2)} m/s`;
  }

  // 4. Height & Roll
  if (data.z !== undefined) {
    const zCm = parseFloat(data.z) * 100.0;
    telemZ.innerHTML = `${zCm.toFixed(1)}<small>cm</small>`;
    const pctZ = Math.max(10, Math.min(100, (zCm / 15.0) * 100));
    meterZ.style.width = `${pctZ}%`;
  }
  if (data.roll !== undefined) {
    telemRoll.textContent = `Roll: ${parseFloat(data.roll).toFixed(1)}°`;
  }
  if (data.step !== undefined) {
    telemStepSub.textContent = `Step: ${data.step} | Surviving`;
  }

  // 5. Status tag
  if (data.status) {
    simStatusTag.textContent = data.status;
  }
}

// Initializations
drawJoystick();
updateWeightUI(300.0);
updateVelocityPreset(0.25);

// Start loops
setInterval(controlLoop, 20);      // 50 Hz control send
setInterval(pollTelemetry, 100);    // 10 Hz telemetry poll
