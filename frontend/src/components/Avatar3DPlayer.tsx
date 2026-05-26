/**
 * Avatar3DPlayer — Live 3D playback of PRE-RENDERED merged GLBs.
 *
 * Plays the ESL_DigiHuman track on top of the looping mixamo.com idle track,
 * crossfading between them. This hides the T-pose moments that can otherwise
 * appear when a new GLB loads.
 */
import { useEffect, useRef, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';


const FADE = 0.25;        // seconds for crossfade between idle and ESL
const POST_HOLD = 0.4;    // seconds to hold the last ESL frame before advancing


function PlayedAvatar({
  glbUrl,
  loopTick,
  onClipEnd,
}: { glbUrl: string; loopTick: number; onClipEnd: () => void }) {
  const { scene, animations } = useGLTF(glbUrl);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const idleActionRef = useRef<THREE.AnimationAction | null>(null);
  const eslActionRef = useRef<THREE.AnimationAction | null>(null);
  const elapsedRef = useRef<number>(0);
  const eslDurationRef = useRef<number>(0);
  const phaseRef = useRef<'fade-in' | 'esl' | 'fade-out' | 'done'>('fade-in');
  const endCalledRef = useRef<boolean>(false);

  useEffect(() => {
    if (!scene) return;
    if (animations.length === 0) {
      const t = setTimeout(onClipEnd, 800);
      return () => clearTimeout(t);
    }
    const mixer = new THREE.AnimationMixer(scene);

    // Pick clips: prefer ESL_DigiHuman, fall back to first; idle is anything else.
    const eslClip = animations.find((c) => c.name === 'ESL_DigiHuman') || animations[0];
    const idleClip = animations.find((c) => c.name !== eslClip.name) || eslClip;

    // Idle loops continuously, full weight at start.
    // Start past frame 0 to skip any T-pose bind frame the Mixamo export embedded.
    const idleAction = mixer.clipAction(idleClip);
    idleAction.setLoop(THREE.LoopRepeat, Infinity);
    idleAction.setEffectiveWeight(1.0);
    idleAction.time = (idleClip.duration || 1.0) * 0.4;
    idleAction.play();
    idleActionRef.current = idleAction;

    // ESL plays once, clamped at last frame; weight starts at 0, paused so it
    // doesn't "leak" into the initial pose.
    const eslAction = mixer.clipAction(eslClip);
    eslAction.setLoop(THREE.LoopOnce, 1);
    eslAction.clampWhenFinished = true;
    eslAction.setEffectiveWeight(0.0);
    eslAction.play();
    eslAction.paused = true;
    eslActionRef.current = eslAction;
    eslDurationRef.current = eslClip.duration || 1.0;

    // Force an immediate mixer update so the very first rendered frame is
    // already in the idle pose, not the raw T-pose bind state.
    mixer.update(0.001);

    mixerRef.current = mixer;
    elapsedRef.current = 0;
    phaseRef.current = 'fade-in';
    endCalledRef.current = false;

    return () => {
      idleAction.stop();
      eslAction.stop();
      mixer.uncacheRoot(scene);
    };
  }, [scene, animations, onClipEnd]);

  // Rewind the playhead in-place when the parent bumps `loopTick`. This keeps
  // the GLB mounted (no flicker / blank frame), and just resets ESL clip time
  // back to 0 so the sign animation replays cleanly.
  useEffect(() => {
    if (!mixerRef.current) return;
    elapsedRef.current = 0;
    endCalledRef.current = false;
    if (eslActionRef.current) {
      eslActionRef.current.time = 0;
      eslActionRef.current.paused = false;
      eslActionRef.current.setEffectiveWeight(0);
    }
    if (idleActionRef.current) {
      idleActionRef.current.setEffectiveWeight(1);
    }
    mixerRef.current.update(0.001);
  }, [loopTick]);

  useFrame((_state, delta) => {
    const mixer = mixerRef.current;
    const idleAction = idleActionRef.current;
    const eslAction = eslActionRef.current;
    if (!mixer || !idleAction || !eslAction) return;

    mixer.update(delta);
    elapsedRef.current += delta;
    const t = elapsedRef.current;
    const eslDur = eslDurationRef.current;

    // Phases:
    //   [0, FADE)              -> fade idle out, ESL in
    //   [FADE, FADE+eslDur)    -> ESL only
    //   [FADE+eslDur, end)     -> hold ESL on its LAST frame (no idle blend-back).
    //
    // We drive the ESL action's playhead MANUALLY each frame so it can never
    // loop and never falls back to the avatar's built-in idle clip (which has
    // its own hand keyframes and was making the sign appear to "play again"
    // after a short pause).
    let idleWeight = 1, eslWeight = 0;
    if (t < FADE) {
      const k = t / FADE;
      idleWeight = 1 - k;
      eslWeight = k;
    } else {
      idleWeight = 0;
      eslWeight = 1;
    }
    // Manual ESL clip-time, clamped to the clip duration so the last frame
    // holds during the POST_HOLD window.
    eslAction.paused = false;
    const clipT = Math.max(0, Math.min(eslDur, t - FADE));
    eslAction.time = clipT;
    idleAction.setEffectiveWeight(idleWeight);
    eslAction.setEffectiveWeight(eslWeight);

    // Advance to next clip only after ESL has played through fully + a short hold.
    // For single-clip playback the parent maps `(idx+1) % length` -> 0 and
    // remounts the same URL, so we guard with endCalledRef to avoid re-firing.
    if (!endCalledRef.current && t >= FADE + eslDur + POST_HOLD) {
      endCalledRef.current = true;
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

  useEffect(() => { setIdx(0); }, [glbUrls?.join(',')]);

  if (!glbUrls || glbUrls.length === 0) return null;
  const displayLabels = labels && labels.length > 0 ? labels : glbUrls.map((u) => {
    const m = u.match(/\/avatar-glb\/([^?]+)/);
    return m ? decodeURIComponent(m[1]) : '?';
  });

  // We keep the GLB mounted across loops to avoid a blank-frame flicker.
  // `loopTick` is just a counter we bump when the playlist wraps around;
  // PlayedAvatar watches it and rewinds the ESL playhead in-place.
  const [loopTick, setLoopTick] = useState(0);
  const currentUrl = glbUrls[idx] + (glbUrls[idx].includes('?') ? '&' : '?') + 'v=1';
  const onClipEnd = () => {
    setIdx((i) => {
      const next = (i + 1) % glbUrls.length;
      if (next <= i) setLoopTick((n) => n + 1);
      return next;
    });
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
          <PlayedAvatar key={currentUrl} glbUrl={currentUrl} loopTick={loopTick} onClipEnd={onClipEnd} />
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

(useGLTF as any).preload = () => {};
