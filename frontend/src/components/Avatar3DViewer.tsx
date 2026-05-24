/**
 * Avatar3DViewer — Clean fresh implementation
 *
 * Principles (verified through landmark visualization):
 * 1. MediaPipe pose_world_landmarks: metric 3D, y=DOWN, origin=hips
 * 2. MediaPipe hand_landmarks: separate coord space, must anchor to wrist
 * 3. Coordinate conversion: GLB = (-x, -y, z)  [mirror X, flip Y]
 * 4. Use Three.js built-in scene graph for FK (no manual matrix math)
 * 5. Position-based IK: rotate each bone to point at target landmark
 */
import React, { useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

export interface RawMocapFrame {
  pose?:  number[][];  // 33 world landmarks [x,y,z,vis]
  rhand?: number[][];  // 21 hand landmarks [x,y,z] (relative)
  lhand?: number[][];  // 21 hand landmarks [x,y,z] (relative)
}
export interface RawMocapData { fps: number; frames: RawMocapFrame[]; }

// ── MediaPipe indices ────────────────────────────────────────────────────────
const MP = {
  NOSE: 0, L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14,
  L_WR: 15, R_WR: 16, L_HP: 23, R_HP: 24,
};

// ── Coordinate conversion ────────────────────────────────────────────────────
// MediaPipe pose_world_landmarks: x=signer-right(negative), y=DOWN(neg=up), z=signer-back
// GLB Three.js (model faces -Z): x=model-right(=screen-left=neg X), y=UP, z=toward-viewer
// → X: signer-right (neg) → model-right (neg) → KEEP X
// → Y: MP y=down → GLB y=up → FLIP Y
// → Z: signer-back (pos MP) → model-back (-Z) → FLIP Z
function mp2glb(lm: number[]): THREE.Vector3 {
  return new THREE.Vector3(lm[0], -lm[1], -lm[2]);
}

function vis(lm: number[]): boolean { return (lm[3] ?? 1.0) > 0.3; }

// ── Position-based IK: rotate bone so its child ends at target ──────────
// Uses the ACTUAL child position (where bone's tip is at rest) as reference direction
function rotateBoneToTarget(
  bone: THREE.Object3D,
  _childPos: THREE.Vector3,
  targetWorldPos: THREE.Vector3,
  restLocalQ: THREE.Quaternion
): THREE.Quaternion {
  bone.updateWorldMatrix(true, false);

  const boneWorldPos = new THREE.Vector3();
  bone.getWorldPosition(boneWorldPos);

  // Find the child bone (the one whose position represents this bone's tip)
  let childBone: THREE.Object3D | undefined;
  for (const child of bone.children) {
    if ((child as any).isBone || child.name) {
      childBone = child;
      break;
    }
  }
  if (!childBone) {
    // No child found, can't IK
    return new THREE.Quaternion();
  }

  // Current child world position (at rest after parent updates)
  const childWorldPos = new THREE.Vector3();
  childBone.getWorldPosition(childWorldPos);

  // Vector from bone start to child (current rest direction in WORLD space)
  const currentWorld = childWorldPos.clone().sub(boneWorldPos).normalize();

  // Desired direction in WORLD space (bone start to target)
  const desiredWorld = targetWorldPos.clone().sub(boneWorldPos).normalize();

  // Rotation in world space from current to desired
  const worldDelta = new THREE.Quaternion().setFromUnitVectors(currentWorld, desiredWorld);

  // Get parent world quaternion to convert world delta to local offset
  const parentWorldQ = new THREE.Quaternion();
  if (bone.parent) bone.parent.getWorldQuaternion(parentWorldQ);

  // To apply worldDelta in world space, we need local offset such that:
  //   parent_world * rest * offset * X_local = worldDelta * parent_world * rest * X_local
  // Solving: offset = (parent_world * rest)⁻¹ * worldDelta * (parent_world * rest)
  //        = rest⁻¹ * parent⁻¹ * worldDelta * parent * rest
  const restInv = restLocalQ.clone().invert();
  const parentInv = parentWorldQ.clone().invert();
  const offset = restInv.clone()
    .multiply(parentInv)
    .multiply(worldDelta)
    .multiply(parentWorldQ)
    .multiply(restLocalQ);

  return offset;
}

// ── Joint angle for fingers ──────────────────────────────────────────────────
function jointAngle(a: number[], o: number[], b: number[]): number {
  const va = new THREE.Vector3(a[0]-o[0], a[1]-o[1], a[2]-o[2]).normalize();
  const vb = new THREE.Vector3(b[0]-o[0], b[1]-o[1], b[2]-o[2]).normalize();
  return Math.max(0, Math.acos(Math.max(-1, Math.min(1, va.dot(vb)))));
}

// ── Landmark Cloud: shows MP 3D points next to avatar ───────────────────────
function LandmarkCloud({ mocapData, timeRef, position, debugRef }: {
  mocapData: RawMocapData;
  timeRef: React.MutableRefObject<number>;
  position: [number,number,number];
  debugRef?: React.MutableRefObject<any>;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const dotsRef  = useRef<THREE.Mesh[]>([]);
  const linesRef = useRef<THREE.Line[]>([]);
  const handDotsRef = useRef<{r: THREE.Mesh[], l: THREE.Mesh[]}>({r:[], l:[]});

  const SKEL_C: [number,number][] = [
    [11,12],[11,13],[13,15],[12,14],[14,16],
    [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
    [0,11],[0,12],
  ];

  useEffect(() => {
    const grp = groupRef.current;
    if (!grp) return;
    while (grp.children.length > 0) grp.remove(grp.children[0]);

    const colors = [
      0xff5566,0xff8844,0xff8844,0xff8844,0xff8844,0xff8844,0xff8844,
      0xffaa44,0xffaa44,0xffaa44,0xffaa44,
      0x66ff99,0x66ff99,
      0x66ccff,0x66ccff,
      0xcc66ff,0xcc66ff,
      0xcccccc,0xcccccc,0xcccccc,0xcccccc,0xcccccc,0xcccccc,
      0x66ff99,0x66ff99,
      0xffcc66,0xffcc66,
      0xffff66,0xffff66,
      0xcccccc,0xcccccc,0xcccccc,0xcccccc,
    ];

    const LM_NAMES = ['NOSE','L_EYE_IN','L_EYE','L_EYE_OUT','R_EYE_IN','R_EYE','R_EYE_OUT',
      'L_EAR','R_EAR','MOUTH_L','MOUTH_R',
      'L_SHOULDER','R_SHOULDER','L_ELBOW','R_ELBOW','L_WRIST','R_WRIST',
      'L_PINKY','R_PINKY','L_INDEX','R_INDEX','L_THUMB','R_THUMB',
      'L_HIP','R_HIP','L_KNEE','R_KNEE','L_ANKLE','R_ANKLE',
      'L_HEEL','R_HEEL','L_FOOT','R_FOOT'];

    const dots: THREE.Mesh[] = [];
    for (let i = 0; i < 33; i++) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.022, 8, 8),
        new THREE.MeshBasicMaterial({ color: colors[i] })
      );
      grp.add(m); dots.push(m);
      // Label sprite
      const canvas = document.createElement('canvas');
      canvas.width = 128; canvas.height = 32;
      const ctx = canvas.getContext('2d')!;
      ctx.font = 'bold 14px sans-serif';
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(0,0,128,32);
      ctx.fillStyle = '#A8FF4B';
      ctx.fillText(LM_NAMES[i], 4, 20);
      const tex = new THREE.CanvasTexture(canvas);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
      sprite.scale.set(0.18, 0.045, 1);
      sprite.userData.lmIdx = i;
      grp.add(sprite);
      (m.userData as any).labelSprite = sprite;
    }
    dotsRef.current = dots;

    const lines: THREE.Line[] = [];
    for (let i = 0; i < SKEL_C.length; i++) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      const ln = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x7c3aed }));
      grp.add(ln); lines.push(ln);
    }
    linesRef.current = lines;

    const rh: THREE.Mesh[] = [], lh: THREE.Mesh[] = [];
    for (let i = 0; i < 21; i++) {
      const rmesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 6, 6),
        new THREE.MeshBasicMaterial({ color: 0xff8844 })
      );
      const lmesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 6, 6),
        new THREE.MeshBasicMaterial({ color: 0x88ff44 })
      );
      grp.add(rmesh); grp.add(lmesh);
      rh.push(rmesh); lh.push(lmesh);
    }
    handDotsRef.current = { r: rh, l: lh };
  }, [mocapData]);

  useFrame(() => {
    if (!mocapData?.frames?.length || !groupRef.current) return;

    const fi = Math.floor(timeRef.current * (mocapData.fps||25)) % mocapData.frames.length;
    const frame = mocapData.frames[fi];
    const pose = frame.pose;
    if (!pose) return;

    // AUTO-SCALE: fit MP landmarks to avatar's body proportions
    const lsh = pose[11], rsh = pose[12], lhp = pose[23], rhp = pose[24];
    const mpShW = Math.abs(lsh[0] - rsh[0]);
    const mpHipY = (lhp[1] + rhp[1]) / 2;
    const mpShY  = (lsh[1] + rsh[1]) / 2;
    const mpTorsoH = Math.abs(mpShY - mpHipY);
    const AV_SH_W = 0.60;
    const AV_TORSO = 0.86;
    const scaleX = mpShW > 0.01 ? AV_SH_W / mpShW : 1;
    const scaleY = mpTorsoH > 0.01 ? AV_TORSO / mpTorsoH : 1;
    const S = (scaleX + scaleY) / 2;

    const mpHipX = (lhp[0] + rhp[0]) / 2;
    const mpHipZ = (lhp[2] + rhp[2]) / 2;
    // Anchor MP hips to avatar hips world (0, 0.034, 0); flip Y/Z via negative scale
    groupRef.current.position.set(
      -mpHipX * S,
      0.034 + mpHipY * S,
      -mpHipZ * S
    );
    groupRef.current.scale.set(S, S, S);

    for (let i = 0; i < 33 && i < pose.length; i++) {
      const lm = pose[i];
      const dot = dotsRef.current[i];
      if (!dot) continue;
      const v = lm[3] ?? 1.0;
      dot.visible = v > 0.3;
      // Centre cloud at hip midpoint, scale to match avatar proportions
      const hipX = ((pose[23][0]+pose[24][0])/2);
      const hipY = ((pose[23][1]+pose[24][1])/2);
      const hipZ = ((pose[23][2]+pose[24][2])/2);
      const S = 1.4;  // approximate scale to match avatar height
      dot.position.set(
        (lm[0] - hipX) * S,
        (-(lm[1] - hipY)) * S,
        (-(lm[2] - hipZ)) * S
      );
      dot.updateMatrixWorld(true);
      // Capture for debug diff
      if (debugRef && [0,11,12,13,14,15,16,23,24].includes(i)) {
        const wp = new THREE.Vector3();
        dot.getWorldPosition(wp);
        debugRef.current.landmarks.set(i, [wp.x, wp.y, wp.z]);
      }
      // Move label sprite next to dot
      const sprite = (dot.userData as any).labelSprite as THREE.Sprite | undefined;
      if (sprite) {
        sprite.visible = dot.visible;
        sprite.position.set((lm[0]-hipX)*S + 0.1, -(lm[1]-hipY)*S, -(lm[2]-hipZ)*S);
      }
    }

    SKEL_C.forEach(([a,b], idx) => {
      const ln = linesRef.current[idx];
      if (!ln) return;
      const pa = pose[a], pb = pose[b];
      if (!pa || !pb) { ln.visible = false; return; }
      const va = pa[3] ?? 1.0, vb = pb[3] ?? 1.0;
      if (va < 0.3 || vb < 0.3) { ln.visible = false; return; }
      ln.visible = true;
      const pos = ln.geometry.attributes.position as THREE.BufferAttribute;
      const hipX = ((pose[23][0]+pose[24][0])/2);
      const hipY = ((pose[23][1]+pose[24][1])/2);
      const hipZ = ((pose[23][2]+pose[24][2])/2);
      const S = 1.4;
      pos.setXYZ(0, (pa[0]-hipX)*S, -(pa[1]-hipY)*S, -(pa[2]-hipZ)*S);
      pos.setXYZ(1, (pb[0]-hipX)*S, -(pb[1]-hipY)*S, -(pb[2]-hipZ)*S);
      pos.needsUpdate = true;
    });

    const renderHand = (
      hand: number[][] | undefined,
      wristIdx: number,
      dots: THREE.Mesh[]
    ) => {
      if (!hand || (Math.abs(hand[0][0])<0.001 && Math.abs(hand[0][1])<0.001)) {
        dots.forEach(d => d.visible = false); return;
      }
      if (!vis(pose[wristIdx])) {
        dots.forEach(d => d.visible = false); return;
      }
      // Use raw pose wrist X/Y; reduce Z scale since pose Z is large depth estimate (-1..0)
      const wx = pose[wristIdx][0], wy = pose[wristIdx][1], wz = pose[wristIdx][2] * 0.1;
      const hw = hand[0];
      for (let i = 0; i < 21 && i < hand.length; i++) {
        const lm = hand[i];
        const d = dots[i];
        if (!d) continue;
        d.visible = true;
        d.position.set(wx + (lm[0]-hw[0]), wy + (lm[1]-hw[1]), wz + (lm[2]-hw[2]));
      }
    };
    renderHand(frame.rhand, MP.R_WR, handDotsRef.current.r);
    renderHand(frame.lhand, MP.L_WR, handDotsRef.current.l);
  });

  // Scale 1.8 to match avatar scale
  // Match avatar scale and position so cloud overlays correctly
  // Avatar scale=1.8, position=[0,-1.8,0]; cloud's origin is MP's hip center
  // Avatar's hips local Y = 1.019, so hips world = -1.8 + 1.019*1.8 = 0.034
  // Match IK target space: no scale (raw MP coords), anchored at avatar hipsWorld (set in useFrame)
  return <group ref={groupRef} />;
}

// ── Avatar component ─────────────────────────────────────────────────────────
function Avatar3D({ mocapData, timeRef, debugRef }: { mocapData: RawMocapData; timeRef: React.MutableRefObject<number>; debugRef?: React.MutableRefObject<any> }) {
  const { scene } = useGLTF('/avatar/arab-man.glb');
  const boneMap   = useRef<Map<string, THREE.Object3D>>(new Map());
  const restMap   = useRef<Map<string, THREE.Quaternion>>(new Map());
  const smoothMap = useRef<Map<string, THREE.Quaternion>>(new Map());
  const hipsRef   = useRef<THREE.Vector3>(new THREE.Vector3());

  useEffect(() => {
    const bm = new Map<string, THREE.Object3D>();
    const rm = new Map<string, THREE.Quaternion>();
    const sm = new Map<string, THREE.Quaternion>();
    scene.traverse(obj => {
      if (!obj.name) return;
      bm.set(obj.name, obj);
      rm.set(obj.name, obj.quaternion.clone());
      sm.set(obj.name, obj.quaternion.clone());
    });
    boneMap.current = bm;
    restMap.current = rm;
    smoothMap.current = sm;
    timeRef.current = 0;

    // Cache hips world position for landmark anchoring
    const hips = bm.get('Hips');
    if (hips) {
      hips.updateWorldMatrix(true, false);
      hips.getWorldPosition(hipsRef.current);
    }
  }, [scene]);

  useEffect(() => { timeRef.current = 0; }, [mocapData]);

  useFrame((_, dt) => {
    if (!mocapData?.frames?.length) return;
    timeRef.current += dt;
    const fi = Math.floor(timeRef.current * (mocapData.fps || 25)) % mocapData.frames.length;
    const frame = mocapData.frames[fi];
    const P = frame.pose;
    if (!P || P.length < 25) return;

    // Reset all bones to rest first
    restMap.current.forEach((q, name) => boneMap.current.get(name)?.quaternion.copy(q));
    scene.updateMatrixWorld(true);

    // Get current hips world position
    const hips = boneMap.current.get('Hips');
    if (!hips) return;
    const hipsWorld = new THREE.Vector3();
    hips.getWorldPosition(hipsWorld);

    // MP landmarks: hip-center, scale 1.4 to match avatar, anchor at avatar hipsWorld
    // Same transform as the landmark cloud display
    const mpHipX = (P[23][0] + P[24][0]) / 2;
    const mpHipY = (P[23][1] + P[24][1]) / 2;
    const mpHipZ = (P[23][2] + P[24][2]) / 2;
    const SCALE = 1.4;
    function mpToWorld(lm: number[]): THREE.Vector3 {
      return new THREE.Vector3(
        (lm[0] - mpHipX) * SCALE,
        -(lm[1] - mpHipY) * SCALE,
        -(lm[2] - mpHipZ) * SCALE,
      ).add(hipsWorld);
    }

    // ── IK: ELBOWS, WRISTS, FINGERS ────────────────────────────────────────────────────
    function applyBone(boneName: string, target: THREE.Vector3) {
      const bone = boneMap.current.get(boneName);
      const rest = restMap.current.get(boneName);
      if (!bone || !rest) return;
      const off = rotateBoneToTarget(bone, new THREE.Vector3(), target, rest);
      const finalQ = rest.clone().multiply(off);
      const sm = smoothMap.current.get(boneName) ?? rest.clone();
      sm.slerp(finalQ, 1.0); // exact match, no smoothing
      smoothMap.current.set(boneName, sm);
      bone.quaternion.copy(sm);
      bone.updateMatrixWorld(true);
    }

    function ikBone(boneName: string, lmIdx: number) {
      if (!vis(P![lmIdx])) return;
      applyBone(boneName, mpToWorld(P![lmIdx]));
    }

    // IK: rotate bones to point at MP landmark world positions
    ikBone('RightArm',      MP.R_EL);
    ikBone('LeftArm',       MP.L_EL);
    ikBone('RightForeArm',  MP.R_WR);
    ikBone('LeftForeArm',   MP.L_WR);

    // Capture bone world positions for debug diff
    if (debugRef) {
      const trackedBones = ['Head','LeftArm','RightArm','LeftForeArm','RightForeArm','LeftHand','RightHand','LeftUpLeg','RightUpLeg'];
      for (const bn of trackedBones) {
        const b = boneMap.current.get(bn);
        if (b) {
          const wp = new THREE.Vector3();
          b.getWorldPosition(wp);
          debugRef.current.bones.set(bn, [wp.x, wp.y, wp.z]);
        }
      }
      debugRef.current.frame = fi;
    }

    // FINGERS: each phalanx targets its actual MP landmark position
    function ikFingers(hand: number[][] | undefined, wristIdx: number, side: 'Right'|'Left') {
      if (!hand || (Math.abs(hand[0][0])<0.001 && Math.abs(hand[0][1])<0.001)) return;
      if (!vis(P![wristIdx])) return;
      const wristWorld = mpToWorld(P![wristIdx]);
      const hw = hand[0];
      const handLmWorld = (i: number) => new THREE.Vector3(
        wristWorld.x + (hand[i][0] - hw[0]),
        wristWorld.y - (hand[i][1] - hw[1]),
        wristWorld.z - (hand[i][2] - hw[2]),
      );
      const CHAINS: [string, number, number, number][] = [
        ['Thumb',  2, 3, 4],
        ['Index',  6, 7, 8],
        ['Middle', 10, 11, 12],
        ['Ring',   14, 15, 16],
        ['Pinky',  18, 19, 20],
      ];
      for (const [name, t1, t2, t3] of CHAINS) {
        if (t3 >= hand.length) continue;
        applyBone(`${side}Hand${name}1`, handLmWorld(t1));
        applyBone(`${side}Hand${name}2`, handLmWorld(t2));
        applyBone(`${side}Hand${name}3`, handLmWorld(t3));
      }
    }
    ikFingers(frame.rhand, MP.R_WR, 'Right');
    ikFingers(frame.lhand, MP.L_WR, 'Left');
  });

  return <primitive object={scene} scale={1.8} position={[0,-1.8,0]} />;
}

// ── Viewer ────────────────────────────────────────────────────────────────────
export function Avatar3DViewer({
  mocapData, className = ''
}: { mocapData: RawMocapData|null; className?: string }) {
  const timeRef = useRef(0);
  // Shared positions store for debug diff capture
  const debugRef = useRef<{
    bones: Map<string, [number,number,number]>,
    landmarks: Map<number, [number,number,number]>,
    frame: number,
  }>({ bones: new Map(), landmarks: new Map(), frame: 0 });
  const [copied, setCopied] = React.useState(false);

  const copyDiff = () => {
    const MP_TO_BONE: Record<number, string> = {
      0: 'Head', 11: 'LeftArm', 12: 'RightArm',
      13: 'LeftForeArm', 14: 'RightForeArm',
      15: 'LeftHand', 16: 'RightHand',
      23: 'LeftUpLeg', 24: 'RightUpLeg',
    };
    const diff: any = { frame: debugRef.current.frame, comparisons: [] };
    for (const [mpIdx, boneName] of Object.entries(MP_TO_BONE)) {
      const lm = debugRef.current.landmarks.get(Number(mpIdx));
      const bone = debugRef.current.bones.get(boneName);
      if (!lm || !bone) continue;
      const dx = lm[0] - bone[0];
      const dy = lm[1] - bone[1];
      const dz = lm[2] - bone[2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      diff.comparisons.push({
        mediapipe: { idx: Number(mpIdx), pos: [+lm[0].toFixed(3), +lm[1].toFixed(3), +lm[2].toFixed(3)] },
        avatar_bone: { name: boneName, pos: [+bone[0].toFixed(3), +bone[1].toFixed(3), +bone[2].toFixed(3)] },
        delta: [+dx.toFixed(3), +dy.toFixed(3), +dz.toFixed(3)],
        distance: +dist.toFixed(3),
      });
    }
    navigator.clipboard.writeText(JSON.stringify(diff, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!mocapData) return null;
  return (
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`}
         style={{ minHeight: 380 }}>
      <Canvas camera={{ position:[0,0.5,4.5], fov:45 }} gl={{ antialias:true }} shadows>
        <ambientLight intensity={0.8} />
        <directionalLight position={[2,4,2]} intensity={1.3} castShadow />
        <pointLight position={[-2,2,-1]} intensity={0.5} color="#7c3aed" />
        <Suspense fallback={null}>
          <Avatar3D mocapData={mocapData} timeRef={timeRef} debugRef={debugRef} />
          {/* Overlay landmark cloud ON the model to spot discrepancies */}
          <LandmarkCloud mocapData={mocapData} timeRef={timeRef} position={[0,0,0]} debugRef={debugRef} />
        </Suspense>
        <OrbitControls enablePan={true} minDistance={1.5} maxDistance={10}
          minPolarAngle={Math.PI/6} maxPolarAngle={Math.PI/1.5}
          target={[0.5,0.6,0]} enableDamping dampingFactor={0.08} />
      </Canvas>
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur
                      text-[#A8FF4B] text-xs font-bold px-4 py-1.5 rounded-full border border-[#A8FF4B]/30">
        🤖 3D Avatar · {mocapData.frames.length} frames
      </div>
      <button onClick={copyDiff}
        className="absolute top-3 right-3 bg-violet-600 hover:bg-violet-500 text-white text-xs
                   font-bold px-3 py-1.5 rounded-lg shadow-lg">
        {copied ? '✓ Copied' : '📋 Copy Diff'}
      </button>
    </div>
  );
}
