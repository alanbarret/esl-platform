/**
 * AvatarViewer — Direct bone manipulation approach
 * Bypasses AnimationMixer entirely — directly sets quaternions on bones
 * every frame based on the animation data from the API.
 */
import React, { useEffect, useRef, Suspense, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import type { GLTFAnimation } from '../types';

// ── Helpers ───────────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function lerpQuat(out: THREE.Quaternion, a: number[], b: number[], t: number) {
  const qa = new THREE.Quaternion(a[0], a[1], a[2], a[3]);
  const qb = new THREE.Quaternion(b[0], b[1], b[2], b[3]);
  out.copy(qa).slerp(qb, t);
}

/** Get interpolated quaternion at time t from a sampler */
function sampleRotation(times: number[], values: number[], t: number): number[] {
  if (times.length === 0) return [0, 0, 0, 1];
  if (t <= times[0]) return values.slice(0, 4);
  if (t >= times[times.length - 1]) return values.slice(-4);

  for (let i = 0; i < times.length - 1; i++) {
    if (t >= times[i] && t < times[i + 1]) {
      const alpha = (t - times[i]) / (times[i + 1] - times[i]);
      const a = values.slice(i * 4, i * 4 + 4);
      const b = values.slice((i + 1) * 4, (i + 1) * 4 + 4);
      const qa = new THREE.Quaternion(a[0], a[1], a[2], a[3]);
      const qb = new THREE.Quaternion(b[0], b[1], b[2], b[3]);
      qa.slerp(qb, alpha);
      return [qa.x, qa.y, qa.z, qa.w];
    }
  }
  return [0, 0, 0, 1];
}

// ── Avatar Component ──────────────────────────────────────────────────────────

interface AvatarProps {
  avatarUrl: string;
  animation: GLTFAnimation | null;
}

function Avatar({ avatarUrl, animation }: AvatarProps) {
  const { scene } = useGLTF(avatarUrl);
  const timeRef = useRef(0);
  const boneMap = useRef<Map<string, THREE.Object3D>>(new Map());
  const restQuat = useRef<Map<string, THREE.Quaternion>>(new Map());

  // Build bone map once on load — target ALL named objects including SkinnedMesh bones
  useEffect(() => {
    const map = new Map<string, THREE.Object3D>();
    const rest = new Map<string, THREE.Quaternion>();

    scene.traverse((obj) => {
      if (obj.name) {
        map.set(obj.name, obj);
        // Store rest quaternion BEFORE any animation
        rest.set(obj.name, obj.quaternion.clone());
      }
    });

    boneMap.current = map;
    restQuat.current = rest;
    timeRef.current = 0;

    const boneNames = [...map.keys()].filter(k =>
      ['Arm','ForeArm','Hand','Shoulder','Spine','Head','Neck','Hips','Thumb','Index','Middle','Ring','Pinky'].some(s => k.includes(s))
    );
    console.log(`[Avatar] Found ${map.size} nodes, ${boneNames.length} signing bones`);
    console.log('[Avatar] Key bones:', boneNames.slice(0, 15).join(', '));
  }, [scene]);

  // Reset timer when animation changes
  useEffect(() => {
    timeRef.current = 0;
  }, [animation]);

  useFrame((_, delta) => {
    if (!animation) return;

    timeRef.current += delta;
    // Loop
    if (timeRef.current > animation.duration) {
      timeRef.current = timeRef.current % animation.duration;
    }

    const t = timeRef.current;

    // Build a map: boneName -> current quaternion at time t
    const boneRots = new Map<string, number[]>();
    for (const channel of animation.channels) {
      const sampler = animation.samplers[channel.sampler];
      const boneName = channel.target.node;
      if (channel.target.path === 'rotation') {
        const rot = sampleRotation(sampler.input, sampler.output, t);
        boneRots.set(boneName, rot);
      }
    }

    // Apply rotations ON TOP of rest pose (multiply, not replace)
    for (const [boneName, rot] of boneRots) {
      const bone = boneMap.current.get(boneName);
      const rest = restQuat.current.get(boneName);
      if (bone && rest) {
        const offset = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]);
        // Multiply: rest * offset (apply offset in bone's local space)
        bone.quaternion.copy(rest).multiply(offset);
      }
    }
  });

  return (
    <group>
      <primitive object={scene} scale={1.8} position={[0, -1.8, 0]} />
    </group>
  );
}

// ── Loading fallback ──────────────────────────────────────────────────────────

function LoadingAvatar() {
  return (
    <mesh>
      <capsuleGeometry args={[0.3, 1.2, 8, 16]} />
      <meshStandardMaterial color="#7c3aed" wireframe />
    </mesh>
  );
}

// ── Viewer ────────────────────────────────────────────────────────────────────

interface AvatarViewerProps {
  avatarUrl?: string;
  animation: GLTFAnimation | null;
  className?: string;
}

export function AvatarViewer({
  avatarUrl = '/avatar/arab-man.glb',
  animation,
  className = '',
}: AvatarViewerProps) {
  return (
    <div
      className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`}
      style={{ minHeight: 420 }}
    >
      <Canvas
        camera={{ position: [0, 0.5, 3.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        shadows
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[2, 4, 2]} intensity={1.3} castShadow />
        <pointLight position={[-2, 2, -2]} intensity={0.5} color="#7c3aed" />
        <Suspense fallback={<LoadingAvatar />}>
          <Avatar avatarUrl={avatarUrl} animation={animation} />
          <Environment preset="studio" />
        </Suspense>
        <OrbitControls
          enablePan={false}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 1.5}
          minDistance={2}
          maxDistance={6}
        />
      </Canvas>

      {animation && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur
                        text-[#A8FF4B] font-bold text-sm px-5 py-2 rounded-full
                        border border-[#A8FF4B]/30 whitespace-nowrap">
          {animation.name.replace(/_/g, ' → ')}
        </div>
      )}
    </div>
  );
}
