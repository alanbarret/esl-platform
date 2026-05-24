/**
 * Avatar3DPlayer — Live 3D playback of PRE-RENDERED merged GLBs.
 *
 * For multi-token sequences (e.g. "SCHOOL CLOSE DOCTOR"), this component plays
 * each token's GLB animation in sequence, auto-advancing when one completes.
 * Each GLB contains the Arab sheikh avatar + the DigiHuman-retargeted animation
 * for that one sign. The browser switches between GLBs and replays — no retargeting
 * happens in the browser.
 */
import React, { useEffect, useRef, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';


function PlayedAvatar({
  glbUrl,
  onClipEnd,
}: { glbUrl: string; onClipEnd: () => void }) {
  const { scene, animations } = useGLTF(glbUrl);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const actionRef = useRef<THREE.AnimationAction | null>(null);
  const durationRef = useRef<number>(0);
  const playedTimeRef = useRef<number>(0);
  const calledEndRef = useRef<boolean>(false);

  useEffect(() => {
    if (!scene || animations.length === 0) {
      // No animation in this GLB → still call onClipEnd after a short pause so the
      // sequence doesn't stall on missing renders.
      const t = setTimeout(onClipEnd, 800);
      return () => clearTimeout(t);
    }
    const mixer = new THREE.AnimationMixer(scene);
    const clip = animations[0];
    const action = mixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
    mixerRef.current = mixer;
    actionRef.current = action;
    durationRef.current = clip.duration || 1.0;
    playedTimeRef.current = 0;
    calledEndRef.current = false;
    return () => {
      action.stop();
      mixer.uncacheRoot(scene);
    };
  }, [scene, animations, onClipEnd]);

  useFrame((_state, delta) => {
    if (!mixerRef.current) return;
    mixerRef.current.update(delta);
    playedTimeRef.current += delta;
    // After the clip's duration completes (+ a small buffer), advance.
    if (!calledEndRef.current && playedTimeRef.current >= durationRef.current + 0.2) {
      calledEndRef.current = true;
      onClipEnd();
    }
  });

  return (
    <group position={[0, -1.0, 0]}>
      <primitive object={scene} />
    </group>
  );
}


export function Avatar3DPlayer({
  glbUrls, labels, className = ''
}: { glbUrls: string[] | null; labels?: string[]; className?: string }) {
  const [idx, setIdx] = useState(0);

  // Reset when the URL list changes
  useEffect(() => { setIdx(0); }, [glbUrls?.join(',')]);

  if (!glbUrls || glbUrls.length === 0) return null;
  const displayLabels = labels && labels.length > 0 ? labels : glbUrls.map((u) => {
    // Fallback: extract the sign name from the URL
    const m = u.match(/\/avatar-glb\/([^?]+)/);
    return m ? decodeURIComponent(m[1]) : '?';
  });

  // Wrap-around the sequence so it loops
  const currentUrl = glbUrls[idx] + (glbUrls[idx].includes('?') ? '&' : '?') + 'v=1';

  const onClipEnd = () => {
    setIdx((i) => (i + 1) % glbUrls.length);
  };

  return (
    <div className={`relative w-full bg-[#09090B] rounded-2xl overflow-hidden ${className}`}
         style={{ minHeight: 380 }}>
      <Canvas
        camera={{ position: [0, 0.4, 1.6], fov: 35 }}
        gl={{ antialias: true }}
        shadows
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[2, 4, 3]} intensity={1.2} castShadow />
        <directionalLight position={[-3, 2, 1]} intensity={0.4} color="#8899ff" />
        <Suspense fallback={null}>
          {/* key on currentUrl so the component fully remounts per token,
              cleanly disposing the previous mixer + scene */}
          <PlayedAvatar key={currentUrl} glbUrl={currentUrl} onClipEnd={onClipEnd} />
        </Suspense>
        <OrbitControls
          enablePan={true}
          minDistance={0.8}
          maxDistance={4}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 1.5}
          target={[0, 0.4, 0]}
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/60 backdrop-blur
                      text-[#A8FF4B] text-xs font-bold px-4 py-1.5 rounded-full border border-[#A8FF4B]/30">
        🤖 3D Avatar · {idx + 1} / {glbUrls.length} · Live playback
      </div>
      {/* Sequence label — shows all signs with the current one highlighted */}
      <div className="absolute top-3 left-3 max-w-[85%] bg-black/70 backdrop-blur px-3 py-1.5
                      rounded-2xl border border-[#A8FF4B]/30 flex flex-wrap gap-1.5 items-center">
        {displayLabels.map((tok, i) => (
          <span key={i}
            className={`text-xs font-bold transition-colors
              ${i === idx ? 'text-[#A8FF4B]' : 'text-gray-400'}`}
            style={{ direction: /[\u0600-\u06ff]/.test(tok) ? 'rtl' : 'ltr' }}>
            {tok}
          </span>
        ))}
      </div>
    </div>
  );
}

// Optional preload helper (no-op safe)
(useGLTF as any).preload = () => {};
