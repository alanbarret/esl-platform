import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

// ── Types ─────────────────────────────────────────────────────────────────────
interface BoneOffset { rx: number; ry: number; rz: number; }
type OffsetMap = Record<string, BoneOffset>;

// ── Bone groups ───────────────────────────────────────────────────────────────
const BONE_GROUPS: Record<string, string[]> = {
  'Spine & Head': ['Hips','Spine','Spine1','Spine2','Neck','Head'],
  'Right Arm':    ['RightShoulder','RightArm','RightForeArm','RightHand'],
  'Left Arm':     ['LeftShoulder','LeftArm','LeftForeArm','LeftHand'],
  'Right Hand':   ['RightHandThumb1','RightHandThumb2','RightHandThumb3',
                   'RightHandIndex1','RightHandIndex2','RightHandIndex3',
                   'RightHandMiddle1','RightHandMiddle2','RightHandMiddle3',
                   'RightHandRing1','RightHandRing2','RightHandRing3',
                   'RightHandPinky1','RightHandPinky2','RightHandPinky3'],
  'Left Hand':    ['LeftHandThumb1','LeftHandThumb2','LeftHandThumb3',
                   'LeftHandIndex1','LeftHandIndex2','LeftHandIndex3',
                   'LeftHandMiddle1','LeftHandMiddle2','LeftHandMiddle3',
                   'LeftHandRing1','LeftHandRing2','LeftHandRing3',
                   'LeftHandPinky1','LeftHandPinky2','LeftHandPinky3'],
  'Legs':         ['LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot'],
};

// ── Avatar component ──────────────────────────────────────────────────────────
interface AvatarProps {
  offsets: OffsetMap;
  onBonesReady: (map: Map<string, THREE.Object3D>, rest: Map<string, THREE.Quaternion>) => void;
}

function Avatar({ offsets, onBonesReady }: AvatarProps) {
  const { scene } = useGLTF('/avatar/arab-man.glb');
  const boneMap = useRef<Map<string, THREE.Object3D>>(new Map());
  const restMap = useRef<Map<string, THREE.Quaternion>>(new Map());
  const readyRef = useRef(false);

  useEffect(() => {
    if (readyRef.current) return;
    readyRef.current = true;
    const bm = new Map<string, THREE.Object3D>();
    const rm = new Map<string, THREE.Quaternion>();
    scene.traverse((obj) => {
      if (obj.name) { bm.set(obj.name, obj); rm.set(obj.name, obj.quaternion.clone()); }
    });
    boneMap.current = bm;
    restMap.current = rm;
    onBonesReady(bm, rm);
  }, [scene, onBonesReady]);

  useFrame(() => {
    for (const [name, off] of Object.entries(offsets)) {
      const bone = boneMap.current.get(name);
      const rest = restMap.current.get(name);
      if (!bone || !rest) continue;
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(off.rx, off.ry, off.rz, 'XYZ'));
      bone.quaternion.copy(rest).multiply(q);
    }
  });

  return (
    <group>
      <primitive object={scene} />
    </group>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function PoseEditor() {
  const [offsets, setOffsets] = useState<OffsetMap>({});
  const [activeGroup, setActiveGroup] = useState('Right Arm');
  const [search, setSearch] = useState('');
  const [showExport, setShowExport] = useState(false);
  const [exportText, setExportText] = useState('');
  const [copied, setCopied] = useState(false);
  const boneMapRef = useRef<Map<string, THREE.Object3D>>(new Map());
  const restMapRef = useRef<Map<string, THREE.Quaternion>>(new Map());

  const onBonesReady = useCallback((bm: Map<string, THREE.Object3D>, rm: Map<string, THREE.Quaternion>) => {
    boneMapRef.current = bm;
    restMapRef.current = rm;
    const init: OffsetMap = {};
    bm.forEach((_, name) => { init[name] = { rx: 0, ry: 0, rz: 0 }; });
    setOffsets(init);
  }, []);

  const setAxis = (bone: string, axis: 'rx'|'ry'|'rz', val: number) => {
    setOffsets(prev => ({ ...prev, [bone]: { ...(prev[bone] || {rx:0,ry:0,rz:0}), [axis]: val } }));
  };

  const resetBone = (bone: string) => {
    setOffsets(prev => ({ ...prev, [bone]: { rx: 0, ry: 0, rz: 0 } }));
  };

  const resetAll = () => {
    setOffsets(prev => {
      const next: OffsetMap = {};
      Object.keys(prev).forEach(k => { next[k] = { rx: 0, ry: 0, rz: 0 }; });
      return next;
    });
  };

  const exportPose = () => {
    const lines = [`"MY_POSE": {`];
    Object.entries(offsets).forEach(([bone, off]) => {
      if (Math.abs(off.rx) > 0.005 || Math.abs(off.ry) > 0.005 || Math.abs(off.rz) > 0.005) {
        lines.push(`    "${bone}": (${off.rx.toFixed(3)}, ${off.ry.toFixed(3)}, ${off.rz.toFixed(3)}),`);
      }
    });
    lines.push(`},`);
    setExportText(lines.join('\n'));
    setShowExport(true);
  };

  const visibleBones = search
    ? Object.keys(offsets).filter(n => n.toLowerCase().includes(search.toLowerCase()))
    : (BONE_GROUPS[activeGroup] || []).filter(n => offsets[n] !== undefined);

  const hasOffset = (bone: string) => {
    const o = offsets[bone];
    return o && (Math.abs(o.rx) > 0.01 || Math.abs(o.ry) > 0.01 || Math.abs(o.rz) > 0.01);
  };

  return (
    <div style={{ height: '100vh', background: '#09090B', color: '#e1e1e1', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ background: '#111', borderBottom: '1px solid #222', padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
        <div style={{ width: 28, height: 28, borderRadius: 6, background: 'linear-gradient(135deg,#7c3aed,#a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14 }}>🤖</div>
        <span style={{ fontWeight: 700, fontSize: 15 }}>ESL Pose Editor</span>
        <span style={{ background: '#7c3aed', color: '#fff', fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 100 }}>3D LIVE</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button onClick={resetAll} style={{ background: '#333', color: '#999', border: '1px solid #444', borderRadius: 7, padding: '6px 12px', fontSize: 12, cursor: 'pointer' }}>↺ Reset All</button>
          <button onClick={exportPose} style={{ background: '#A8FF4B', color: '#000', border: 'none', borderRadius: 7, padding: '6px 14px', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}>⬇ Export Pose</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', flex: 1, overflow: 'hidden' }}>
        {/* 3D Canvas */}
        <Canvas
          camera={{ position: [0, 1.3, 2.8], fov: 45 }}
          gl={{ antialias: true }}
          shadows
          style={{ background: '#09090B' }}
        >
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 4, 2]} intensity={1.3} castShadow />
          <pointLight position={[-2, 2, -2]} intensity={0.5} color="#8060ff" />
          <mesh rotation={[-Math.PI/2, 0, 0]} receiveShadow>
            <planeGeometry args={[6, 6]} />
            <meshStandardMaterial color="#111120" roughness={0.9} />
          </mesh>
          <gridHelper args={[4, 20, '#1a1a2e', '#1a1a2e']} />
          <React.Suspense fallback={null}>
            <Avatar offsets={offsets} onBonesReady={onBonesReady} />
          </React.Suspense>
          <OrbitControls target={[0, 1.0, 0]} enableDamping dampingFactor={0.08} minDistance={0.5} maxDistance={6} />
        </Canvas>

        {/* Sidebar */}
        <div style={{ background: '#111', borderLeft: '1px solid #1e1e1e', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Search */}
          <div style={{ padding: '8px 12px', borderBottom: '1px solid #1e1e1e' }}>
            <input
              value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search bones..."
              style={{ width: '100%', background: '#1a1a2e', border: '1px solid #333', borderRadius: 6, padding: '6px 10px', color: '#fff', fontSize: 12, outline: 'none' }}
            />
          </div>

          {/* Group tabs */}
          {!search && (
            <div style={{ display: 'flex', gap: 4, padding: '6px 12px', borderBottom: '1px solid #1e1e1e', flexWrap: 'wrap' }}>
              {Object.keys(BONE_GROUPS).map(g => (
                <button key={g} onClick={() => setActiveGroup(g)}
                  style={{ background: activeGroup === g ? '#7c3aed' : '#1a1a1a', border: `1px solid ${activeGroup === g ? '#7c3aed' : '#333'}`, borderRadius: 100, padding: '3px 10px', fontSize: 10, color: activeGroup === g ? '#fff' : '#888', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                  {g}
                </button>
              ))}
            </div>
          )}

          {/* Bone list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
            {visibleBones.map(name => {
              const off = offsets[name] || { rx: 0, ry: 0, rz: 0 };
              const active = hasOffset(name);
              return (
                <div key={name} style={{ background: active ? '#1a1020' : '#181818', border: `1px solid ${active ? '#7c3aed44' : '#222'}`, borderRadius: 8, padding: '8px 10px', marginBottom: 3 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: active ? '#c4b5fd' : '#8b8b8b' }}>{name}</span>
                    <button onClick={() => resetBone(name)} style={{ fontSize: 9, background: '#222', border: 'none', color: '#666', borderRadius: 3, padding: '1px 5px', cursor: 'pointer' }}>reset</button>
                  </div>
                  {(['rx','ry','rz'] as const).map((ax, i) => (
                    <div key={ax} style={{ display: 'grid', gridTemplateColumns: '14px 1fr 38px', alignItems: 'center', gap: 4, marginBottom: 3 }}>
                      <span style={{ fontSize: 9, fontWeight: 700, textAlign: 'center', color: ['#ef4444','#22c55e','#3b82f6'][i] }}>{ax.toUpperCase()}</span>
                      <input type="range" min={-3.14} max={3.14} step={0.01} value={off[ax]}
                        onChange={e => setAxis(name, ax, parseFloat(e.target.value))}
                        style={{ width: '100%', height: 3, accentColor: ['#ef4444','#22c55e','#3b82f6'][i], cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: 9, fontFamily: 'monospace', color: '#999', textAlign: 'right' }}>{off[ax].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Export modal */}
      {showExport && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#111', border: '1px solid #333', borderRadius: 12, padding: 20, width: 500, maxHeight: '70vh', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700 }}>Export Pose → demo_server.py</h2>
            <textarea readOnly value={exportText}
              style={{ background: '#0a0a0a', border: '1px solid #333', borderRadius: 6, padding: 10, color: '#A8FF4B', fontSize: 11, fontFamily: 'monospace', flex: 1, resize: 'none', minHeight: 200, outline: 'none' }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setShowExport(false)} style={{ background: '#333', border: 'none', color: '#999', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12 }}>Close</button>
              <button onClick={() => { navigator.clipboard.writeText(exportText); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
                style={{ background: '#7c3aed', border: 'none', color: '#fff', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12, fontWeight: 700 }}>
                {copied ? '✓ Copied!' : 'Copy to Clipboard'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
