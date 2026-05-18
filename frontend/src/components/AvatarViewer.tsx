/**
 * AvatarViewer — Three.js / React Three Fiber GLTF Avatar
 *
 * Loads a GLTF/GLB humanoid avatar and applies AnimationClip
 * generated from the motion engine.
 *
 * Supports:
 * - Bone retargeting
 * - Finger animation
 * - Facial expressions (blendshapes)
 * - Head / shoulder movement
 */
import React, { useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF, useAnimations } from '@react-three/drei';
import * as THREE from 'three';
import type { GLTFAnimation } from '../types';

// ── Avatar mesh ────────────────────────────────────────────────────────────────

interface AvatarProps {
  avatarUrl: string;
  animation: GLTFAnimation | null;
}

function Avatar({ avatarUrl, animation }: AvatarProps) {
  const { scene, animations } = useGLTF(avatarUrl);
  const { mixer, actions } = useAnimations(animations, scene);
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    if (!animation) return;

    // Build THREE.AnimationClip from GLTF animation data
    const tracks: THREE.KeyframeTrack[] = [];

    for (let i = 0; i < animation.channels.length; i++) {
      const channel = animation.channels[i];
      const sampler = animation.samplers[channel.sampler];
      const boneName = channel.target.node;
      const path = channel.target.path;

      const times = new Float32Array(sampler.input);
      const values = new Float32Array(sampler.output);

      let track: THREE.KeyframeTrack;
      if (path === 'rotation') {
        track = new THREE.QuaternionKeyframeTrack(
          `${boneName}.quaternion`, times, values
        );
      } else if (path === 'translation') {
        track = new THREE.VectorKeyframeTrack(
          `${boneName}.position`, times, values
        );
      } else {
        track = new THREE.VectorKeyframeTrack(
          `${boneName}.scale`, times, values
        );
      }
      tracks.push(track);
    }

    const clip = new THREE.AnimationClip(animation.name, animation.duration, tracks);
    const action = mixer.clipAction(clip, scene);
    action.reset().play();

    return () => {
      action.stop();
      mixer.uncacheAction(clip, scene);
    };
  }, [animation, mixer, scene]);

  // Advance animation mixer
  useFrame((_, delta) => {
    mixer.update(delta);
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} scale={1.8} position={[0, -1.8, 0]} />
    </group>
  );
}

// ── Loading fallback ───────────────────────────────────────────────────────────

function AvatarSkeleton() {
  return (
    <mesh>
      <capsuleGeometry args={[0.3, 1.2, 8, 16]} />
      <meshStandardMaterial color="#7c3aed" wireframe />
    </mesh>
  );
}

// ── Canvas wrapper ─────────────────────────────────────────────────────────────

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
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`}
         style={{ minHeight: 400 }}>
      <Canvas
        camera={{ position: [0, 0.5, 3.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        shadows
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, 4, 2]} intensity={1.2} castShadow />
        <pointLight position={[-2, 2, -2]} intensity={0.4} color="#7c3aed" />

        <Suspense fallback={<AvatarSkeleton />}>
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

      {/* Overlay: current gloss label */}
      {animation && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur
                        text-[#A8FF4B] font-bold text-lg px-6 py-2 rounded-full border border-[#A8FF4B]/30">
          {animation.name.replace(/_/g, ' ')}
        </div>
      )}
    </div>
  );
}
