const videoElement = document.getElementById('input_video');
const canvas2d = document.getElementById('canvas2d');
const ctx2d = canvas2d.getContext('2d');
const canvas3d = document.getElementById('canvas3d');
const statusText = document.getElementById('status');

// --- 1. THREE.JS SCENE SETUP ---
const scene = new THREE.Scene();
const camera3d = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
camera3d.position.z = 5;

// Function to create particle systems
function createCursedEffect(color, size) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(2000 * 3);
    for(let i=0; i<6000; i++) pos[i] = (Math.random() - 0.5) * 2;
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size: size, color: color, transparent: true, blending: THREE.AdditiveBlending });
    const points = new THREE.Points(geo, mat);
    points.visible = false;
    scene.add(points);
    return points;
}

const purpleOrb = createCursedEffect(0x8000ff, 0.08); // Hollow Purple
const redOrb = createCursedEffect(0xff0000, 0.12);    // Cursed Technique: Red

// --- 2. THE PRECISION MATH ---
function mapCoordinates(landmark) {
    // Correcting for aspect ratio and FOV
    const x = (landmark.x * 2 - 1) * (window.innerWidth / window.innerHeight) * 3.5;
    const y = -(landmark.y * 2 - 1) * 3.5;
    return { x, y };
}

// --- 3. GESTURE DETECTION ---
function getActiveTechnique(landmarks) {
    const thumb = landmarks[4];
    const index = landmarks[8];
    const middle = landmarks[12];
    const ring = landmarks[16];
    
    // 1. Detect Purple (Pinch)
    const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    if (pinchDist < 0.03) return "PURPLE";

    // 2. Detect Red (Index Pointing Up)
    // Check if index tip is significantly higher than index knuckle AND middle finger is curled
    const isIndexUp = index.y < landmarks[6].y;
    const isMiddleDown = middle.y > landmarks[10].y;
    const isRingDown = ring.y > landmarks[14].y;
    
    if (isIndexUp && isMiddleDown && isRingDown) return "RED";

    return "NONE";
}

// --- 4. PROCESSING ---
const hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.8, minTrackingConfidence: 0.8 });

hands.onResults((results) => {
    canvas2d.width = window.innerWidth;
    canvas2d.height = window.innerHeight;
    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
    ctx2d.drawImage(results.image, 0, 0, canvas2d.width, canvas2d.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        const technique = getActiveTechnique(landmarks);
        const pos = mapCoordinates(landmarks[8]); // Track Index Tip

        // Reset visibility
        purpleOrb.visible = false;
        redOrb.visible = false;

        if (technique === "PURPLE") {
            statusText.innerText = "Hollow Purple 🟣";
            purpleOrb.visible = true;
            purpleOrb.position.set(pos.x, pos.y, 0);
            purpleOrb.rotation.z += 0.2; // Fast chaotic spin
            purpleOrb.scale.setScalar(Math.sin(Date.now() * 0.02) * 0.5 + 1.2);
        } 
        else if (technique === "RED") {
            statusText.innerText = "Cursed Technique: Red 🔴";
            redOrb.visible = true;
            redOrb.position.set(pos.x, pos.y, 0);
            // Repulsion animation: particles "explode" outwards
            redOrb.scale.setScalar(Math.random() * 0.5 + 1.5);
            redOrb.rotation.y += 0.5;
        } else {
            statusText.innerText = "Idle";
        }
    }
    renderer.render(scene, camera3d);
});

const camera = new Camera(videoElement, {
    onFrame: async () => { await hands.send({ image: videoElement }); },
    width: 1280, height: 720
});
camera.start();