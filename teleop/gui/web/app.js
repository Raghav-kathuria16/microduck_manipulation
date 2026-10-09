// ── State ──────────────────────────────────────────────────────────
const state = {
    // Navigation (click-to-navigate)
    nav_target: null,  // {x, y} in world coords, or null
    nav_stop: false,
    
    // Head aim (right joystick)
    head_pitch: 0.0,
    head_yaw: 0.0,
    
    // Action buttons
    actions: {
        grasp: false,
        pick: false,
        sit: false,
        reset_head: false,
        randomize: false
    }
};

// Robot pose (received from server)
let robotPose = { x: 0, y: 0, yaw: 0, vx: 0, wz: 0 };

// ── Apartment map bounds ──────────────────────────────────────────
const MAP = {
    xMin: -4.5, xMax: 4.5,
    yMin: -3.5, yMax: 3.5,
    // Room layout for drawing (simplified)
    walls: [
        // Outer walls
        {x1: -4, y1: -3, x2: 4, y2: -3},   // bottom
        {x1: -4, y1: 3, x2: 4, y2: 3},     // top
        {x1: -4, y1: -3, x2: -4, y2: 3},   // left
        {x1: 4, y1: -3, x2: 4, y2: 3},     // right
        // Interior walls
        {x1: -1, y1: -3, x2: -1, y2: 1},   // living room right wall
        {x1: -1, y1: 1, x2: -1, y2: 3},    // kitchen bottom
        {x1: 0.5, y1: -3, x2: 0.5, y2: 3}, // corridor right
        {x1: 0.5, y1: 1, x2: 4, y2: 1},    // bedroom/office divider
        {x1: 0.5, y1: -1, x2: 4, y2: -1},  // office/bathroom divider
    ],
    rooms: [
        {name: 'Living', cx: -2.5, cy: -1},
        {name: 'Kitchen', cx: -2.5, cy: 2},
        {name: 'Corridor', cx: -0.25, cy: 0},
        {name: 'Bedroom', cx: 2.25, cy: 2},
        {name: 'Office', cx: 2.25, cy: 0},
        {name: 'Bath', cx: 2.25, cy: -2},
    ]
};

// ── UI Elements ───────────────────────────────────────────────────
const statusEl = document.getElementById('status');
const navStatusText = document.getElementById('nav_status_text');
const valVx = document.getElementById('val_vx');
const valWz = document.getElementById('val_wz');
const valHead = document.getElementById('val_head');
const valRx = document.getElementById('val_rx');
const valRy = document.getElementById('val_ry');
const valYaw = document.getElementById('val_yaw');

const canvas = document.getElementById('nav_map');
const ctx = canvas.getContext('2d');

// ── WebSocket ─────────────────────────────────────────────────────
let ws = null;
const WS_URL = `ws://${window.location.hostname}:8765`;

function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('[WS] Connected');
        statusEl.textContent = 'Connected';
        statusEl.className = 'status connected';
    };
    
    ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        statusEl.className = 'status disconnected';
        setTimeout(connectWebSocket, 1000);
    };
    
    ws.onmessage = (evt) => {
        // Receive robot pose updates from server (response to each command)
        try {
            const data = JSON.parse(evt.data);
            if (data.type === 'pose') {
                robotPose.x = data.x;
                robotPose.y = data.y;
                robotPose.yaw = data.yaw;
                robotPose.vx = data.vx || 0;
                robotPose.wz = data.wz || 0;
                
                // Update nav status
                if (data.nav_status === 'navigating') {
                    navStatusText.textContent = 'Navigating...';
                    navStatusText.style.color = '#34d399';
                } else if (data.nav_status === 'arrived') {
                    navStatusText.textContent = '✓ Arrived';
                    navStatusText.style.color = '#38bdf8';
                    state.nav_target = null;  // Clear target on map
                }
                
                updateTelemetry();
            }
        } catch(e) {}
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

connectWebSocket();

// ── Send state at 30Hz ────────────────────────────────────────────
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const msg = {
            type: 'command',
            nav_target: state.nav_target,
            nav_stop: state.nav_stop,
            head_pitch: state.head_pitch,
            head_yaw: state.head_yaw,
            actions: state.actions,
        };
        ws.send(JSON.stringify(msg));
        
        // Reset one-shot actions after sending
        state.nav_stop = false;
        state.actions.pick = false;
        state.actions.reset_head = false;
        state.actions.randomize = false;
    }
}, 1000 / 30);

// ── Canvas coordinate conversion ──────────────────────────────────
function worldToCanvas(wx, wy) {
    const px = ((wx - MAP.xMin) / (MAP.xMax - MAP.xMin)) * canvas.width;
    const py = ((MAP.yMax - wy) / (MAP.yMax - MAP.yMin)) * canvas.height;  // flip Y
    return [px, py];
}

function canvasToWorld(cx, cy) {
    const wx = MAP.xMin + (cx / canvas.width) * (MAP.xMax - MAP.xMin);
    const wy = MAP.yMax - (cy / canvas.height) * (MAP.yMax - MAP.yMin);  // flip Y
    return [wx, wy];
}

// ── Draw the map ──────────────────────────────────────────────────
function drawMap() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let x = MAP.xMin; x <= MAP.xMax; x += 1) {
        const [px] = worldToCanvas(x, 0);
        ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height); ctx.stroke();
    }
    for (let y = MAP.yMin; y <= MAP.yMax; y += 1) {
        const [, py] = worldToCanvas(0, y);
        ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(canvas.width, py); ctx.stroke();
    }
    
    // Room labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    MAP.rooms.forEach(r => {
        const [px, py] = worldToCanvas(r.cx, r.cy);
        ctx.fillText(r.name, px, py);
    });
    
    // Walls
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 2;
    MAP.walls.forEach(w => {
        const [x1, y1] = worldToCanvas(w.x1, w.y1);
        const [x2, y2] = worldToCanvas(w.x2, w.y2);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });
    
    // Navigation target (if set)
    if (state.nav_target) {
        const [tx, ty] = worldToCanvas(state.nav_target.x, state.nav_target.y);
        
        // Pulsing target circle
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
        ctx.beginPath();
        ctx.arc(tx, ty, 8 + pulse * 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(239, 68, 68, ${0.3 + pulse * 0.2})`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
        
        // Crosshair
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(tx - 12, ty); ctx.lineTo(tx + 12, ty); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(tx, ty - 12); ctx.lineTo(tx, ty + 12); ctx.stroke();
    }
    
    // Robot (arrow showing position + heading)
    const [rx, ry] = worldToCanvas(robotPose.x, robotPose.y);
    const yaw = -robotPose.yaw;  // canvas Y is flipped
    
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(yaw);
    
    // Robot body
    ctx.beginPath();
    ctx.moveTo(12, 0);       // nose
    ctx.lineTo(-8, -7);      // left
    ctx.lineTo(-4, 0);       // indent
    ctx.lineTo(-8, 7);       // right
    ctx.closePath();
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.strokeStyle = '#0ea5e9';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.restore();
    
    requestAnimationFrame(drawMap);
}

drawMap();

// ── Map click handler ─────────────────────────────────────────────
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
    const [wx, wy] = canvasToWorld(cx, cy);
    
    console.log(`[NAV] Click → world (${wx.toFixed(2)}, ${wy.toFixed(2)})`);
    state.nav_target = { x: wx, y: wy };
    navStatusText.textContent = `→ (${wx.toFixed(1)}, ${wy.toFixed(1)})`;
    navStatusText.style.color = '#34d399';
});

// ── Stop button ───────────────────────────────────────────────────
document.getElementById('btn_stop').addEventListener('click', () => {
    state.nav_target = null;
    state.nav_stop = true;
    navStatusText.textContent = 'Stopped';
    navStatusText.style.color = '#f87171';
    setTimeout(() => {
        navStatusText.textContent = 'Idle';
        navStatusText.style.color = '#94a3b8';
    }, 1500);
});

// ── Head aim joystick (right) ─────────────────────────────────────
const optionsRight = {
    zone: document.getElementById('zone_joystick_right'),
    mode: 'static',
    position: { left: '50%', top: '50%' },
    color: '#f472b6',
    size: 150
};

const joyRight = nipplejs.create(optionsRight);

joyRight.on('move', (evt, data) => {
    const max_aim = 1.0;
    const distance = data.distance / 75.0;
    const angle = data.angle.radian;
    
    state.head_pitch = Math.sin(angle) * distance * max_aim;
    state.head_yaw = -Math.cos(angle) * distance * max_aim;
    
    updateTelemetry();
});

joyRight.on('end', () => {
    state.head_pitch = 0.0;
    state.head_yaw = 0.0;
    updateTelemetry();
});

// ── Telemetry ─────────────────────────────────────────────────────
function updateTelemetry() {
    valRx.textContent = robotPose.x.toFixed(2);
    valRy.textContent = robotPose.y.toFixed(2);
    valYaw.textContent = (robotPose.yaw * 180 / Math.PI).toFixed(1);
    valVx.textContent = robotPose.vx.toFixed(2);
    valWz.textContent = robotPose.wz.toFixed(2);
    valHead.textContent = `${state.head_pitch.toFixed(2)}, ${state.head_yaw.toFixed(2)}`;
}

// ── Action buttons ────────────────────────────────────────────────
document.getElementById('btn_grasp').addEventListener('click', () => {
    state.actions.grasp = !state.actions.grasp;
    document.getElementById('btn_grasp').style.filter = state.actions.grasp ? 'brightness(1.5)' : 'none';
});
document.getElementById('btn_pick').addEventListener('click', () => { state.actions.pick = true; });
document.getElementById('btn_sit').addEventListener('click', () => { state.actions.sit = !state.actions.sit; });
document.getElementById('btn_reset_head').addEventListener('click', () => { state.actions.reset_head = true; });
document.getElementById('btn_randomize').addEventListener('click', () => { state.actions.randomize = true; });
