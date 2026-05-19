/**
 * Avatar3DViewer — 3D GLB model driven by MediaPipe mocap frames
 * 
 * BONE MAPPING ANALYSIS (from GLB skeleton study):
 * - Model: Ready Player Me / Wolf3D rig, Y-up, bones along local Y-axis
 * - RightArm rest quaternion: [0.4767, -0.0426, 0.0586, 0.8761]
 *   local_Y = (-0.143, +0.539, +0.830) → arm hangs down-forward in rest
 * - MediaPipe: normalized image coords, Z=depth (negative=toward camera), mirrored
 * 
 * RETARGETING STRATEGY:
 * 1. Compute MediaPipe bone direction vectors in 3D world space
 * 2. Find rotation from GLB rest direction to target direction
 * 3. Apply as local offset quaternion: bone.q = rest_q * offset_q
 */
import React, { useEffect, useRef, useCallback, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
// Raw mocap format (full 33 pose landmarks + 21 hand landmarks per frame)
export interface RawMocapFrame {
  pose?: number[][];  // 33 landmarks [x,y,z,visibility]
  rhand?: number[][]; // 21 landmarks [x,y,z]
  lhand?: number[][]; // 21 landmarks [x,y,z]
}
export interface RawMocapData { fps: number; frames: RawMocapFrame[]; }

// ── MediaPipe landmark indices ────────────────────────────────────────────────
const MP = {
  NOSE:0, L_EYE:2, R_EYE:5, L_EAR:7, R_EAR:8,
  L_SH:11, R_SH:12, L_EL:13, R_EL:14, L_WR:15, R_WR:16,
  L_HP:23, R_HP:24, L_KN:25, R_KN:26, L_AN:27, R_AN:28,
  // Hand landmarks (21 per hand)
  H_WR:0, H_THUMB_CMC:1, H_THUMB_MCP:2, H_THUMB_IP:3, H_THUMB_TIP:4,
  H_INDEX_MCP:5, H_INDEX_PIP:6, H_INDEX_DIP:7, H_INDEX_TIP:8,
  H_MID_MCP:9, H_MID_PIP:10, H_MID_DIP:11, H_MID_TIP:12,
  H_RING_MCP:13, H_RING_PIP:14, H_RING_DIP:15, H_RING_TIP:16,
  H_PINKY_MCP:17, H_PINKY_PIP:18, H_PINKY_DIP:19, H_PINKY_TIP:20,
};

// ── GLB rest quaternions (from skeleton analysis) ─────────────────────────────
const REST_Q: Record<string, [number,number,number,number]> = {
  RightShoulder: [0.5049,-0.4895,0.5147,0.4904],
  RightArm:      [0.4767,-0.0426,0.0586,0.8761],
  RightForeArm:  [-0.0496,-0.0233,-0.2205,0.9738],
  RightHand:     [0.0462,-0.0387,0.0185,0.998],
  LeftShoulder:  [0.5049,0.4895,-0.5147,0.4904],
  LeftArm:       [0.4767,0.0426,-0.0586,0.8761],
  LeftForeArm:   [-0.0496,0.0233,0.2205,0.9738],
  LeftHand:      [0.0462,0.0387,-0.0185,0.998],
  Spine1:        [-0.0322,0,0,0.9995],
  Spine2:        [0.0472,0,0,0.9989],
  Neck:          [0.2031,0,0,0.9792],
};

// ── Math helpers ─────────────────────────────────────────────────────────────
function normV(v: THREE.Vector3) { return v.clone().normalize(); }

function mpLm(lm: number[], scale=1): THREE.Vector3 {
  // MediaPipe → Three.js: flip X (mirrored), Y stays, negate Z (depth)
  return new THREE.Vector3(-lm[0]*scale, -lm[1]*scale, lm[2] ? lm[2]*scale : 0);
}

function rotationFromTo(from: THREE.Vector3, to: THREE.Vector3): THREE.Quaternion {
  const f = normV(from), t = normV(to);
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(f, t);
  return q;
}

function angleCurl(a: number[], o: number[], b: number[]): number {
  const A = new THREE.Vector3(-a[0],-a[1],a[2]||0);
  const O = new THREE.Vector3(-o[0],-o[1],o[2]||0);
  const B = new THREE.Vector3(-b[0],-b[1],b[2]||0);
  const va = A.clone().sub(O).normalize();
  const vb = B.clone().sub(O).normalize();
  return Math.max(0, Math.PI - Math.acos(Math.max(-1, Math.min(1, va.dot(vb)))));
}

// ── Retargeting: MediaPipe frame → bone quaternion offsets ───────────────────
function frameToOffsets(frame: RawMocapFrame): Map<string, THREE.Quaternion> {
  const out = new Map<string, THREE.Quaternion>();
  const pose = frame.pose;
  const rh = frame.rhand;
  const lh = frame.lhand;

  if (pose && pose.length >= 29) {
    // ── Spine lean ──────────────────────────────────────────────────────────
    const lsh = mpLm(pose[MP.L_SH]), rsh = mpLm(pose[MP.R_SH]);
    const lhp = mpLm(pose[MP.L_HP]), rhp = mpLm(pose[MP.R_HP]);
    const shMid = lsh.clone().add(rsh).multiplyScalar(0.5);
    const hpMid = lhp.clone().add(rhp).multiplyScalar(0.5);
    const spineDir = shMid.clone().sub(hpMid).normalize();

    // Spine tilt: angle from world Y
    const spineRx = Math.atan2(spineDir.z, spineDir.y) * 0.35;
    const spineRz = Math.atan2(shMid.x - hpMid.x, Math.abs(shMid.y - hpMid.y)) * 0.4;
    out.set('Spine1', new THREE.Quaternion().setFromEuler(new THREE.Euler(spineRx*0.5, 0, spineRz*0.5, 'XYZ')));
    out.set('Spine2', new THREE.Quaternion().setFromEuler(new THREE.Euler(spineRx*0.5, 0, spineRz*0.5, 'XYZ')));

    // ── Right arm chain ─────────────────────────────────────────────────────
    const rShP = mpLm(pose[MP.R_SH]);
    const rElP = mpLm(pose[MP.R_EL]);
    const rWrP = mpLm(pose[MP.R_WR]);

    if (pose[MP.R_SH][3] > 0.3 && pose[MP.R_EL][3] > 0.3) {
      // Direction from shoulder to elbow in MediaPipe space
      const mpArmDir = rElP.clone().sub(rShP).normalize();

      // GLB RightArm rest: local Y = (-0.143, +0.539, +0.830)
      const restArmDir = new THREE.Vector3(-0.143, 0.539, 0.830);
      const armOffset = rotationFromTo(restArmDir, mpArmDir);
      // Scale down to avoid over-rotation
      armOffset.slerp(new THREE.Quaternion(), 0.3);
      out.set('RightArm', armOffset);

      if (pose[MP.R_WR][3] > 0.3) {
        const mpForeDir = rWrP.clone().sub(rElP).normalize();
        // GLB RightForeArm rest local Y = (0.432, 0.898, -0.086)
        const restForeDir = new THREE.Vector3(0.432, 0.898, -0.086);
        const curl = angleCurl(pose[MP.R_SH]!, pose[MP.R_EL]!, pose[MP.R_WR]!);
        const foreOffset = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(0, 0, -1), -curl * 0.5
        );
        out.set('RightForeArm', foreOffset);
      }
    }

    // ── Left arm chain ──────────────────────────────────────────────────────
    const lShP = mpLm(pose[MP.L_SH]);
    const lElP = mpLm(pose[MP.L_EL]);
    const lWrP = mpLm(pose[MP.L_WR]);

    if (pose[MP.L_SH][3] > 0.3 && pose[MP.L_EL][3] > 0.3) {
      const mpArmDir = lElP.clone().sub(lShP).normalize();
      // GLB LeftArm rest local Y = (0.143, 0.539, 0.830) — mirror of right
      const restArmDir = new THREE.Vector3(0.143, 0.539, 0.830);
      const armOffset = rotationFromTo(restArmDir, mpArmDir);
      armOffset.slerp(new THREE.Quaternion(), 0.3);
      out.set('LeftArm', armOffset);

      if (pose[MP.L_WR][3] > 0.3) {
        const curl = angleCurl(pose[MP.L_SH]!, pose[MP.L_EL]!, pose[MP.L_WR]!);
        const foreOffset = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(0, 0, 1), -curl * 0.5
        );
        out.set('LeftForeArm', foreOffset);
      }
    }
  }

  // ── Finger retargeting ────────────────────────────────────────────────────
  const fingerChains = [
    ['Thumb',  [MP.H_THUMB_CMC, MP.H_THUMB_MCP, MP.H_THUMB_IP,  MP.H_THUMB_TIP]],
    ['Index',  [MP.H_INDEX_MCP, MP.H_INDEX_PIP, MP.H_INDEX_DIP, MP.H_INDEX_TIP]],
    ['Middle', [MP.H_MID_MCP,   MP.H_MID_PIP,   MP.H_MID_DIP,   MP.H_MID_TIP]],
    ['Ring',   [MP.H_RING_MCP,  MP.H_RING_PIP,  MP.H_RING_DIP,  MP.H_RING_TIP]],
    ['Pinky',  [MP.H_PINKY_MCP, MP.H_PINKY_PIP, MP.H_PINKY_DIP, MP.H_PINKY_TIP]],
  ] as [string, number[]][];

  function retargetFingers(hand: number[][], side: 'Right'|'Left') {
    const sign = side === 'Right' ? 1 : -1;
    for (const [finger, [mcp, pip, dip, tip]] of fingerChains) {
      // Curl = how bent is each joint
      const c1 = Math.max(0, (Math.PI - angleCurl(hand[mcp], hand[pip], hand[dip])) * 0.9);
      const c2 = Math.max(0, (Math.PI - angleCurl(hand[pip], hand[dip], hand[tip])) * 0.7);
      // Apply curl as rotation around local Z (finger curl axis)
      out.set(`${side}Hand${finger}1`, new THREE.Quaternion().setFromEuler(new THREE.Euler(c1*sign, 0, 0)));
      out.set(`${side}Hand${finger}2`, new THREE.Quaternion().setFromEuler(new THREE.Euler(c2*sign, 0, 0)));
      out.set(`${side}Hand${finger}3`, new THREE.Quaternion().setFromEuler(new THREE.Euler(c2*0.6*sign, 0, 0)));
    }
    // Wrist orientation
    const wv = new THREE.Vector3(
      -(hand[MP.H_MID_MCP][0] - hand[MP.H_WR][0]),
      -(hand[MP.H_MID_MCP][1] - hand[MP.H_WR][1]),
      0
    ).normalize();
    const wristTwist = Math.atan2(wv.x, Math.abs(wv.y)) * 0.3 * sign;
    out.set(`${side}Hand`, new THREE.Quaternion().setFromEuler(new THREE.Euler(0, wristTwist, 0)));
  }

  if (rh && !(Math.abs(rh[0][0]) < 0.001 && Math.abs(rh[0][1]) < 0.001)) {
    retargetFingers(rh, 'Right');
  }
  if (lh && !(Math.abs(lh[0][0]) < 0.001 && Math.abs(lh[0][1]) < 0.001)) {
    retargetFingers(lh, 'Left');
  }

  return out;
}

// ── Avatar 3D component ───────────────────────────────────────────────────────
interface Avatar3DProps {
  mocapData: RawMocapData;
}

function Avatar3D({ mocapData }: Avatar3DProps) {
  const { scene } = useGLTF('/avatar/arab-man.glb');
  const boneMap = useRef<Map<string, THREE.Object3D>>(new Map());
  const restMap = useRef<Map<string, THREE.Quaternion>>(new Map());
  const smoothMap = useRef<Map<string, THREE.Quaternion>>(new Map());
  const timeRef = useRef(0);

  useEffect(() => {
    const bm = new Map<string, THREE.Object3D>();
    const rm = new Map<string, THREE.Quaternion>();
    const sm = new Map<string, THREE.Quaternion>();
    scene.traverse(obj => {
      if (obj.name) {
        bm.set(obj.name, obj);
        rm.set(obj.name, obj.quaternion.clone());
        sm.set(obj.name, obj.quaternion.clone());
      }
    });
    boneMap.current = bm; restMap.current = rm; smoothMap.current = sm;
    timeRef.current = 0;
  }, [scene]);

  useEffect(() => { timeRef.current = 0; }, [mocapData]);

  useFrame((_, delta) => {
    if (!mocapData?.frames?.length) return;
    timeRef.current += delta;
    const fi = Math.floor(timeRef.current * (mocapData.fps||25)) % mocapData.frames.length;
    const frame = mocapData.frames[fi] as RawMocapFrame;
    const offsets = frameToOffsets(frame);

    // Reset all bones to rest first
    restMap.current.forEach((q, name) => {
      const bone = boneMap.current.get(name);
      if (bone) bone.quaternion.copy(q);
    });

    // Apply offsets with smooth interpolation
    for (const [name, offset] of offsets) {
      const bone = boneMap.current.get(name);
      const rest = restMap.current.get(name);
      if (!bone || !rest) continue;
      const target = new THREE.Quaternion().copy(rest).multiply(offset);
      const smooth = smoothMap.current.get(name) ?? rest.clone();
      smooth.slerp(target, 0.3);
      smoothMap.current.set(name, smooth);
      bone.quaternion.copy(smooth);
    }
  });

  return <primitive object={scene} scale={1.8} position={[0,-1.8,0]} />;
}

// ── Main viewer ───────────────────────────────────────────────────────────────
interface Avatar3DViewerProps {
  mocapData: RawMocapData | null;
  className?: string;
}

export function Avatar3DViewer({ mocapData, className='' }: Avatar3DViewerProps) {
  if (!mocapData) return null;

  return (
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`}
         style={{minHeight: 380}}>
      <Canvas camera={{position:[0,0.5,3.2], fov:42}} gl={{antialias:true}} shadows>
        <ambientLight intensity={0.8} />
        <directionalLight position={[2,4,2]} intensity={1.3} castShadow />
        <pointLight position={[-2,2,-1]} intensity={0.5} color="#7c3aed" />
        <Suspense fallback={null}>
          <Avatar3D mocapData={mocapData} />
        </Suspense>
        <OrbitControls enablePan={false} minDistance={1.5} maxDistance={5}
          minPolarAngle={Math.PI/6} maxPolarAngle={Math.PI/1.5}
          target={[0,0.2,0]} enableDamping dampingFactor={0.08} />
      </Canvas>
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur
                      text-[#A8FF4B] text-xs font-bold px-4 py-1.5 rounded-full border border-[#A8FF4B]/30">
        🤖 3D Avatar · {mocapData.frames.length} frames
      </div>
    </div>
  );
}
