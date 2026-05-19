/**
 * AvatarViewer — Dual-mode avatar driver
 * Mode 1: GLTFAnimation (keyframe bone quaternions) — used for static SIGN_POSES
 * Mode 2: MocapFrames (live MediaPipe landmarks) — 1:1 match with source skeleton
 */
import React, { useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import type { GLTFAnimation } from '../types';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface MocapFrame {
  pose?: {
    lsh: number[]; rsh: number[]; lel: number[]; rel: number[];
    lwr: number[]; rwr: number[]; lhp: number[]; rhp: number[];
    lkn: number[]; rkn: number[]; lan: number[]; ran: number[];
    nose: number[]; lvis: number; rvis: number;
  };
  rhand?: number[][];
  lhand?: number[][];
}
export interface MocapData { fps: number; frames: MocapFrame[]; }

// ── Helpers ───────────────────────────────────────────────────────────────────
function sampleRotation(times: number[], values: number[], t: number): number[] {
  if (!times.length) return [0,0,0,1];
  if (t <= times[0]) return values.slice(0,4);
  if (t >= times[times.length-1]) return values.slice(-4);
  for (let i = 0; i < times.length-1; i++) {
    if (t >= times[i] && t < times[i+1]) {
      const alpha = (t-times[i])/(times[i+1]-times[i]);
      const qa = new THREE.Quaternion(...(values.slice(i*4,i*4+4) as [number,number,number,number]));
      const qb = new THREE.Quaternion(...(values.slice((i+1)*4,(i+1)*4+4) as [number,number,number,number]));
      return [qa.slerp(qb,alpha).x, qa.y, qa.z, qa.w];
    }
  }
  return [0,0,0,1];
}

function e2q(rx: number, ry: number, rz: number): THREE.Quaternion {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
}

function vn(a: number[], b: number[]): number[] {
  const d = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
  const l = Math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) + 1e-12;
  return [d[0]/l, d[1]/l, d[2]/l];
}
function ang(a: number[], o: number[], b: number[]): number {
  const va = vn(o,a), vb = vn(o,b);
  return Math.acos(Math.max(-1, Math.min(1, va[0]*vb[0]+va[1]*vb[1]+va[2]*vb[2])));
}

// ── Landmark → bone offset mapping ───────────────────────────────────────────
function landmarksToOffsets(frame: MocapFrame): Map<string, THREE.Quaternion> {
  const out = new Map<string, THREE.Quaternion>();
  const { pose, rhand, lhand } = frame;

  if (pose) {
    const { lsh, rsh, lel, rel, lwr, rwr, lhp, rhp, lvis, rvis } = pose;
    const shm = [(lsh[0]+rsh[0])/2, (lsh[1]+rsh[1])/2, (lsh[1]+rsh[2])/2];
    const hpm = [(lhp[0]+rhp[0])/2, (lhp[1]+rhp[1])/2, (lhp[2]+rhp[2])/2];
    const sv = vn(hpm, shm);

    out.set('Spine1', e2q(sv[2]*0.3, 0, Math.atan2(shm[0]-hpm[0], Math.abs(shm[1]-hpm[1]))*0.3));
    out.set('Spine2', e2q(sv[2]*0.4, 0, Math.atan2(shm[0]-hpm[0], Math.abs(shm[1]-hpm[1]))*0.4));

    if (rvis > 0.3) {
      const av = vn(rsh, rel);
      out.set('RightArm', e2q(0, Math.atan2(av[0],av[1])*0.5, -Math.atan2(-av[2],av[1])*0.9));
      out.set('RightForeArm', e2q(0, 0, -(Math.PI - ang(rsh,rel,rwr))*0.6));
    }
    if (lvis > 0.3) {
      const av = vn(lsh, lel);
      out.set('LeftArm', e2q(0, -Math.atan2(av[0],av[1])*0.5, Math.atan2(-av[2],av[1])*0.9));
      out.set('LeftForeArm', e2q(0, 0, -(Math.PI - ang(lsh,lel,lwr))*0.6));
    }
  }

  // Finger curl helper
  const W=0,TC=1,TM=2,TI=3,TT=4,IM=5,IP=6,ID=7,IT=8,MM=9,MP=10,MD=11,MT=12,RM=13,RP=14,RD=15,RT=16,PM=17,PP=18,PD=19,PT=20;
  function fingerCurl(h: number[][], mcp: number, pip: number, dip: number, tip: number) {
    const c1 = Math.max(0, (Math.PI - ang(h[mcp],h[pip],h[dip]))*0.9);
    const c2 = Math.max(0, (Math.PI - ang(h[pip],h[dip],h[tip]))*0.7);
    return [c1, c2];
  }

  if (rhand) {
    const wv = vn(rhand[W], rhand[MM]);
    out.set('RightHand', e2q(0, -Math.atan2(wv[0], Math.abs(wv[1]))*0.3, 0));
    for (const [bp,mcp,pip,dip,tip] of [['RightHandIndex',IM,IP,ID,IT],['RightHandMiddle',MM,MP,MD,MT],['RightHandRing',RM,RP,RD,RT],['RightHandPinky',PM,PP,PD,PT]] as [string,number,number,number,number][]) {
      const [c1,c2] = fingerCurl(rhand,mcp,pip,dip,tip);
      out.set(bp+'1', e2q(c1,0,0)); out.set(bp+'2', e2q(c2,0,0)); out.set(bp+'3', e2q(c2*0.6,0,0));
    }
    const ta = ang(rhand[TC],rhand[TM],rhand[TI]), tb = ang(rhand[TM],rhand[TI],rhand[TT]);
    out.set('RightHandThumb1', e2q(Math.max(0,(Math.PI-ta)*0.6),0,-0.3));
    out.set('RightHandThumb2', e2q(Math.max(0,(Math.PI-tb)*0.5),0,0));
  }
  if (lhand) {
    const wv = vn(lhand[W], lhand[MM]);
    out.set('LeftHand', e2q(0, Math.atan2(wv[0], Math.abs(wv[1]))*0.3, 0));
    for (const [bp,mcp,pip,dip,tip] of [['LeftHandIndex',IM,IP,ID,IT],['LeftHandMiddle',MM,MP,MD,MT],['LeftHandRing',RM,RP,RD,RT],['LeftHandPinky',PM,PP,PD,PT]] as [string,number,number,number,number][]) {
      const [c1,c2] = fingerCurl(lhand,mcp,pip,dip,tip);
      out.set(bp+'1', e2q(c1,0,0)); out.set(bp+'2', e2q(c2,0,0)); out.set(bp+'3', e2q(c2*0.6,0,0));
    }
    const ta = ang(lhand[TC],lhand[TM],lhand[TI]), tb = ang(lhand[TM],lhand[TI],lhand[TT]);
    out.set('LeftHandThumb1', e2q(Math.max(0,(Math.PI-ta)*0.6),0,0.3));
    out.set('LeftHandThumb2', e2q(Math.max(0,(Math.PI-tb)*0.5),0,0));
  }
  return out;
}

// ── Avatar Component ──────────────────────────────────────────────────────────
interface AvatarProps {
  avatarUrl: string;
  animation: GLTFAnimation | null;
  mocapData: MocapData | null;
}

function Avatar({ avatarUrl, animation, mocapData }: AvatarProps) {
  const { scene } = useGLTF(avatarUrl);
  const timeRef = useRef(0);
  const frameRef = useRef(0);
  const boneMap = useRef<Map<string, THREE.Object3D>>(new Map());
  const restQuat = useRef<Map<string, THREE.Quaternion>>(new Map());
  const smoothQ = useRef<Map<string, THREE.Quaternion>>(new Map());

  useEffect(() => {
    const map = new Map<string, THREE.Object3D>();
    const rest = new Map<string, THREE.Quaternion>();
    const smooth = new Map<string, THREE.Quaternion>();
    scene.traverse((obj) => {
      if (obj.name) {
        map.set(obj.name, obj);
        rest.set(obj.name, obj.quaternion.clone());
        smooth.set(obj.name, obj.quaternion.clone());
      }
    });
    boneMap.current = map; restQuat.current = rest; smoothQ.current = smooth;
    timeRef.current = 0; frameRef.current = 0;
  }, [scene]);

  useEffect(() => { timeRef.current = 0; frameRef.current = 0; }, [animation, mocapData]);

  useFrame((_, delta) => {
    // ── Mode 2: MoCap landmark-driven ───────────────────────────────────────
    if (mocapData && mocapData.frames.length > 0) {
      const fps = mocapData.fps || 25;
      timeRef.current += delta;
      const fi = Math.floor(timeRef.current * fps) % mocapData.frames.length;
      const frame = mocapData.frames[fi];
      const offsets = landmarksToOffsets(frame);

      for (const [name, obj] of boneMap.current) {
        const rest = restQuat.current.get(name)!;
        const target = offsets.has(name)
          ? new THREE.Quaternion().copy(rest).multiply(offsets.get(name)!)
          : rest.clone();

        // Smooth 0.25 alpha
        const cur = smoothQ.current.get(name) ?? rest.clone();
        cur.slerp(target, 0.25);
        smoothQ.current.set(name, cur);
        obj.quaternion.copy(cur);
      }
      return;
    }

    // ── Mode 1: GLTF keyframe animation ─────────────────────────────────────
    if (!animation) return;
    timeRef.current += delta;
    if (timeRef.current > animation.duration) timeRef.current %= animation.duration;
    const t = timeRef.current;

    const boneRots = new Map<string, number[]>();
    for (const ch of animation.channels) {
      const s = animation.samplers[ch.sampler];
      if (ch.target.path === 'rotation')
        boneRots.set(ch.target.node, sampleRotation(s.input, s.output, t));
    }
    for (const [name, rot] of boneRots) {
      const bone = boneMap.current.get(name);
      const rest = restQuat.current.get(name);
      if (bone && rest) {
        const offset = new THREE.Quaternion(rot[0],rot[1],rot[2],rot[3]);
        bone.quaternion.copy(rest).multiply(offset);
      }
    }
  });

  return <group><primitive object={scene} scale={1.8} position={[0,-1.8,0]} /></group>;
}

function LoadingAvatar() {
  return (
    <mesh>
      <capsuleGeometry args={[0.3,1.2,8,16]} />
      <meshStandardMaterial color="#7c3aed" wireframe />
    </mesh>
  );
}

// ── Viewer ────────────────────────────────────────────────────────────────────
interface AvatarViewerProps {
  avatarUrl?: string;
  animation: GLTFAnimation | null;
  mocapData?: MocapData | null;
  className?: string;
}

export function AvatarViewer({ avatarUrl='/avatar/arab-man.glb', animation, mocapData=null, className='' }: AvatarViewerProps) {
  const label = mocapData ? '🎯 Live MoCap' : animation ? animation.name.replace(/_/g,' → ') : null;
  return (
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`} style={{minHeight:420}}>
      <Canvas camera={{position:[0,0.5,3.5],fov:45}} gl={{antialias:true,alpha:true}} shadows>
        <ambientLight intensity={0.7} />
        <directionalLight position={[2,4,2]} intensity={1.3} castShadow />
        <pointLight position={[-2,2,-2]} intensity={0.5} color="#7c3aed" />
        <Suspense fallback={<LoadingAvatar />}>
          <Avatar avatarUrl={avatarUrl} animation={animation} mocapData={mocapData} />
          <Environment preset="studio" />
        </Suspense>
        <OrbitControls enablePan={false} minPolarAngle={Math.PI/6} maxPolarAngle={Math.PI/1.5} minDistance={2} maxDistance={6} />
      </Canvas>
      {label && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur text-[#A8FF4B] font-bold text-sm px-5 py-2 rounded-full border border-[#A8FF4B]/30 whitespace-nowrap">
          {label}
        </div>
      )}
    </div>
  );
}
