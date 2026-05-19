/**
 * Avatar3DViewer — 3D GLB model driven by MediaPipe pose_world_landmarks
 *
 * KEY INSIGHT: pose_world_landmarks (not pose_landmarks) gives:
 * - Real 3D metric coordinates (meters)
 * - Origin at hip center
 * - Y-UP coordinate system (same as GLB)
 * - Z = depth (32% contribution vs 10% for image landmarks)
 * - Upper arm ~22-25cm (realistic)
 *
 * RETARGETING:
 * - World landmarks are in GLB-compatible coordinate space (Y-up)
 * - MediaPipe world: x=right, y=UP, z=toward-camera (right-handed)
 * - GLB Three.js: x=right, y=UP, z=toward-viewer
 * - Conversion: GLB_z = -MP_z (depth direction flip only)
 *
 * BONE MAPPING: offset = conj(boneWorldQ) * worldCorrection * boneWorldQ
 * (Proven 0° error formula from skeleton study)
 */
import React, { useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

export interface RawMocapFrame {
  pose?: number[][];   // 33 world landmarks [x, y, z, visibility] - metric space, Y-up
  rhand?: number[][];  // 21 hand landmarks [x, y, z]
  lhand?: number[][];  // 21 hand landmarks [x, y, z]
  world_landmarks?: boolean;
}
export interface RawMocapData { fps: number; frames: RawMocapFrame[]; world_landmarks?: boolean; }

// ── MediaPipe landmark indices ─────────────────────────────────────────────────
const L_SH=11, R_SH=12, L_EL=13, R_EL=14, L_WR=15, R_WR=16;
const L_HP=23, R_HP=24;

// ── Accumulated world quaternions for key bones (from skeleton analysis) ──────
// Chain: Hips(-7.9°) * Spine(-3.7°) * Spine1(-3.7°) * Spine2(+5.4°) = small net tilt
// RightShoulder world q (accumulated through full chain):
const R_SH_WORLD = new THREE.Quaternion(0.4862,-0.4698,0.5327,0.5090); // full chain
const R_ARM_REST = new THREE.Quaternion(0.4767,-0.0426,0.0586,0.8761);
const R_FORE_REST= new THREE.Quaternion(-0.0496,-0.0233,-0.2205,0.9738);
const L_SH_WORLD = new THREE.Quaternion(0.4862,0.4698,-0.5327,0.5090); // mirror
const L_ARM_REST = new THREE.Quaternion(0.4767,0.0426,-0.0586,0.8761);
const L_FORE_REST= new THREE.Quaternion(-0.0496,0.0233,0.2205,0.9738);

// ── Convert MediaPipe world landmark to Three.js/GLB space ────────────────────
// MP world: x=right, y=UP, z=toward-camera  →  GLB: x=right, y=UP, z=toward-viewer
// Only need to flip Z (depth direction)
function mpWorld(lm: number[]): THREE.Vector3 {
  return new THREE.Vector3(lm[0], lm[1], -lm[2]);
}

function vis(lm: number[]): number { return lm[3] ?? 1.0; }

// ── Correct retargeting: 0° error formula ────────────────────────────────────
// offset = conj(boneWorldQ) * worldCorrection * boneWorldQ
// worldCorrection = rotationFromTo(currentBoneWorldDir, targetDir)
function retargetBone(targetDir: THREE.Vector3, boneWorldQ: THREE.Quaternion): THREE.Quaternion {
  const currentDir = new THREE.Vector3(0,1,0).applyQuaternion(boneWorldQ).normalize();
  const worldCorr  = new THREE.Quaternion().setFromUnitVectors(currentDir, targetDir.clone().normalize());
  return boneWorldQ.clone().invert().multiply(worldCorr).multiply(boneWorldQ);
}

// ── Joint angle helpers ───────────────────────────────────────────────────────
function jointAngle(a: THREE.Vector3, o: THREE.Vector3, b: THREE.Vector3): number {
  const va = a.clone().sub(o).normalize();
  const vb = b.clone().sub(o).normalize();
  return Math.acos(Math.max(-1, Math.min(1, va.dot(vb))));
}

function curlAngle(a: number[], o: number[], b: number[]): number {
  const va = new THREE.Vector3(a[0]-o[0], a[1]-o[1], -(a[2]-o[2])).normalize();
  const vb = new THREE.Vector3(b[0]-o[0], b[1]-o[1], -(b[2]-o[2])).normalize();
  return Math.max(0, Math.acos(Math.max(-1, Math.min(1, va.dot(vb)))));
}

// ── Main retargeting ──────────────────────────────────────────────────────────
function frameToOffsets(frame: RawMocapFrame): Map<string, THREE.Quaternion> {
  const out = new Map<string, THREE.Quaternion>();
  const pose = frame.pose;
  const rh = frame.rhand;
  const lh = frame.lhand;

  if (pose && pose.length >= 25) {
    // ── Spine ──────────────────────────────────────────────────────────────
    const shMid = mpWorld(pose[L_SH]).add(mpWorld(pose[R_SH])).multiplyScalar(0.5);
    const hpMid = mpWorld(pose[L_HP]).add(mpWorld(pose[R_HP])).multiplyScalar(0.5);
    const spineDir = shMid.clone().sub(hpMid).normalize();
    // GLB spine points up (+Y), offset = deviation from pure up
    const spineOffset = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0,1,0), spineDir
    );
    // Scale to 30% — spine doesn't need to fully match (torso crop)
    out.set('Spine1', new THREE.Quaternion().slerp(spineOffset, 0.3));
    out.set('Spine2', new THREE.Quaternion().slerp(spineOffset, 0.3));

    // ── Right arm ─────────────────────────────────────────────────────────
    if (vis(pose[R_SH]) > 0.3 && vis(pose[R_EL]) > 0.3) {
      const rShW = mpWorld(pose[R_SH]);
      const rElW = mpWorld(pose[R_EL]);
      const rWrW = mpWorld(pose[R_WR]);

      // Upper arm direction in world space (true 3D from world landmarks)
      const armDir = rElW.clone().sub(rShW).normalize();

      // boneWorldQ = accumulated world rotation of RightArm at rest
      const rArmWorldQ = R_SH_WORLD.clone().multiply(R_ARM_REST);
      const armOffset  = retargetBone(armDir, rArmWorldQ);
      const armBlended = new THREE.Quaternion().slerp(armOffset, 0.8);
      out.set('RightArm', armBlended);

      if (vis(pose[R_WR]) > 0.3) {
        const foreDir = rWrW.clone().sub(rElW).normalize();
        // ForeArm world = sh_world * (arm_rest * arm_offset) * fore_rest
        const rArmFinal   = R_ARM_REST.clone().multiply(armBlended);
        const rForeWorldQ = R_SH_WORLD.clone().multiply(rArmFinal).multiply(R_FORE_REST);
        const foreOffset  = retargetBone(foreDir, rForeWorldQ);
        const foreBlended = new THREE.Quaternion().slerp(foreOffset, 0.7);
        out.set('RightForeArm', foreBlended);

        // Wrist from hand landmarks
        if (rh && !(Math.abs(rh[0][0])<0.001 && Math.abs(rh[0][1])<0.001)) {
          const idxDir = new THREE.Vector3(rh[5][0]-rh[0][0], rh[5][1]-rh[0][1], -(rh[5][2]-rh[0][2]));
          const pnkDir = new THREE.Vector3(rh[17][0]-rh[0][0], rh[17][1]-rh[0][1], -(rh[17][2]-rh[0][2]));
          const palmNorm = new THREE.Vector3().crossVectors(idxDir, pnkDir).normalize();
          const rForeW2  = R_SH_WORLD.clone().multiply(rArmFinal).multiply(R_FORE_REST).multiply(foreBlended);
          out.set('RightHand', new THREE.Quaternion().slerp(retargetBone(palmNorm, rForeW2), 0.4));
        }
      }
    }

    // ── Left arm ──────────────────────────────────────────────────────────
    if (vis(pose[L_SH]) > 0.3 && vis(pose[L_EL]) > 0.3) {
      const lShW = mpWorld(pose[L_SH]);
      const lElW = mpWorld(pose[L_EL]);
      const lWrW = mpWorld(pose[L_WR]);

      const armDir     = lElW.clone().sub(lShW).normalize();
      const lArmWorldQ = L_SH_WORLD.clone().multiply(L_ARM_REST);
      const armOffset  = retargetBone(armDir, lArmWorldQ);
      const armBlended = new THREE.Quaternion().slerp(armOffset, 0.8);
      out.set('LeftArm', armBlended);

      if (vis(pose[L_WR]) > 0.3) {
        const foreDir    = lWrW.clone().sub(lElW).normalize();
        const lArmFinal  = L_ARM_REST.clone().multiply(armBlended);
        const lForeWorldQ= L_SH_WORLD.clone().multiply(lArmFinal).multiply(L_FORE_REST);
        const foreOffset = retargetBone(foreDir, lForeWorldQ);
        const foreBlended= new THREE.Quaternion().slerp(foreOffset, 0.7);
        out.set('LeftForeArm', foreBlended);

        if (lh && !(Math.abs(lh[0][0])<0.001 && Math.abs(lh[0][1])<0.001)) {
          const idxDir = new THREE.Vector3(lh[5][0]-lh[0][0], lh[5][1]-lh[0][1], -(lh[5][2]-lh[0][2]));
          const pnkDir = new THREE.Vector3(lh[17][0]-lh[0][0], lh[17][1]-lh[0][1], -(lh[17][2]-lh[0][2]));
          const palmNorm = new THREE.Vector3().crossVectors(pnkDir, idxDir).normalize(); // flipped for left
          const lForeW2  = L_SH_WORLD.clone().multiply(lArmFinal).multiply(L_FORE_REST).multiply(foreBlended);
          out.set('LeftHand', new THREE.Quaternion().slerp(retargetBone(palmNorm, lForeW2), 0.4));
        }
      }
    }
  }

  // ── Fingers — coordinate-independent joint angles ─────────────────────────
  const CHAINS: [string,number,number,number,number][] = [
    ['Thumb',1,2,3,4],['Index',5,6,7,8],['Middle',9,10,11,12],
    ['Ring',13,14,15,16],['Pinky',17,18,19,20],
  ];

  function retargetFingers(hand: number[][], side: 'Right'|'Left') {
    const sign = side==='Right' ? -1 : 1;
    for (const [name,mcp,pip,dip,tip] of CHAINS) {
      // Use 3D angle (Z flipped to GLB space)
      const c1 = Math.max(0, curlAngle(hand[mcp],hand[pip],hand[dip])) * 0.85;
      const c2 = Math.max(0, curlAngle(hand[pip],hand[dip],hand[tip])) * 0.75;
      out.set(`${side}Hand${name}1`, new THREE.Quaternion().setFromEuler(new THREE.Euler(sign*c1,0,0)));
      out.set(`${side}Hand${name}2`, new THREE.Quaternion().setFromEuler(new THREE.Euler(sign*c2,0,0)));
      out.set(`${side}Hand${name}3`, new THREE.Quaternion().setFromEuler(new THREE.Euler(sign*c2*0.5,0,0)));
    }
  }

  const rhValid = rh && !(Math.abs(rh[0][0])<0.001 && Math.abs(rh[0][1])<0.001);
  const lhValid = lh && !(Math.abs(lh[0][0])<0.001 && Math.abs(lh[0][1])<0.001);
  if (rhValid) retargetFingers(rh!, 'Right');
  if (lhValid) retargetFingers(lh!, 'Left');

  return out;
}

// ── Avatar 3D component ───────────────────────────────────────────────────────
interface Avatar3DProps { mocapData: RawMocapData; }

function Avatar3D({ mocapData }: Avatar3DProps) {
  const { scene }   = useGLTF('/avatar/arab-man.glb');
  const boneMap     = useRef<Map<string,THREE.Object3D>>(new Map());
  const restMap     = useRef<Map<string,THREE.Quaternion>>(new Map());
  const smoothMap   = useRef<Map<string,THREE.Quaternion>>(new Map());
  const timeRef     = useRef(0);

  useEffect(() => {
    const bm=new Map<string,THREE.Object3D>(), rm=new Map<string,THREE.Quaternion>(), sm=new Map<string,THREE.Quaternion>();
    scene.traverse(obj => {
      if(obj.name){ bm.set(obj.name,obj); rm.set(obj.name,obj.quaternion.clone()); sm.set(obj.name,obj.quaternion.clone()); }
    });
    boneMap.current=bm; restMap.current=rm; smoothMap.current=sm; timeRef.current=0;
  }, [scene]);

  useEffect(() => { timeRef.current=0; }, [mocapData]);

  useFrame((_,delta) => {
    if (!mocapData?.frames?.length) return;
    timeRef.current += delta;
    const fi = Math.floor(timeRef.current*(mocapData.fps||25)) % mocapData.frames.length;
    const offsets = frameToOffsets(mocapData.frames[fi]);
    restMap.current.forEach((q,name) => { boneMap.current.get(name)?.quaternion.copy(q); });
    for (const [name,offset] of offsets) {
      const bone=boneMap.current.get(name), rest=restMap.current.get(name);
      if(!bone||!rest) continue;
      const target = rest.clone().multiply(offset);
      const smooth = smoothMap.current.get(name) ?? rest.clone();
      smooth.slerp(target, 0.3);
      smoothMap.current.set(name,smooth);
      bone.quaternion.copy(smooth);
    }
  });

  return <primitive object={scene} scale={1.8} position={[0,-1.8,0]} />;
}

// ── Viewer ────────────────────────────────────────────────────────────────────
interface Avatar3DViewerProps { mocapData: RawMocapData|null; className?: string; }

export function Avatar3DViewer({ mocapData, className='' }: Avatar3DViewerProps) {
  if (!mocapData) return null;
  return (
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`} style={{minHeight:380}}>
      <Canvas camera={{position:[0,0.5,3.2],fov:42}} gl={{antialias:true}} shadows>
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
        🤖 3D Avatar · world landmarks · {mocapData.frames.length} frames
      </div>
    </div>
  );
}
