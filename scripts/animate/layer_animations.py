#!/usr/bin/env python3
"""Layer two GLB animation files into one, with bone-level filtering.

Strategy:
  - Take a "base" animation (e.g. idle) and an "overlay" animation (e.g. ESL signing).
  - For bones in the OVERRIDE set: use the overlay's tracks (resampled/looped to the
    base duration if needed). For bones NOT in OVERRIDE: keep the base's tracks.
  - Output a single combined GLB animation file (containing just the tracks +
    auxiliary "named bone" nodes) suitable for feeding into merge_animation.py.

Usage:
  python3 layer_animations.py <base.glb> <overlay.glb> <output.glb>
                              [--override "LeftShoulder,LeftArm,..."]

Default OVERRIDE = upper body + hands (everything an ESL retarget produces).
"""

import sys, json, struct, argparse
from pathlib import Path
import numpy as np
from pygltflib import GLTF2


# Default: anything an ESL retarget would produce
DEFAULT_OVERRIDE_PREFIXES = (
    'Spine', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
)


def in_override(name: str, override_prefixes) -> bool:
    return any(name.startswith(p) for p in override_prefixes)


def read_accessor_floats(gltf: GLTF2, accessor_idx: int) -> np.ndarray:
    """Read accessor data as a float32 array, shape inferred from .type."""
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    n = acc.count
    type_components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc.type]
    arr = np.frombuffer(blob, dtype=np.float32, count=n * type_components, offset=offset)
    return arr.reshape(n, type_components) if type_components > 1 else arr.copy()


def get_anim_tracks(gltf: GLTF2):
    """Return: {bone_name: {'rotation': (times, quats), 'translation': (...), 'scale': (...)}}"""
    out = {}
    name_by_idx = {i: n.name for i, n in enumerate(gltf.nodes) if n.name}
    for anim in gltf.animations:
        for ch in anim.channels:
            bone_name = name_by_idx.get(ch.target.node)
            if not bone_name:
                continue
            sampler = anim.samplers[ch.sampler]
            times = read_accessor_floats(gltf, sampler.input)
            vals = read_accessor_floats(gltf, sampler.output)
            out.setdefault(bone_name, {})[ch.target.path] = (times, vals)
    return out


def resample(times_target: np.ndarray, times_src: np.ndarray, vals_src: np.ndarray) -> np.ndarray:
    """Linear resample for translation/scale; nearest-neighbor for rotation (good enough for short clips).
    For proper rotation we'd slerp; left as nearest for simplicity since we mostly use this for
    matching base duration to overlay duration (1:1)."""
    if len(times_src) == 1:
        return np.broadcast_to(vals_src, (len(times_target),) + vals_src.shape[1:]).copy()
    # Use np.interp per component
    out = np.zeros((len(times_target),) + vals_src.shape[1:], dtype=np.float32)
    if vals_src.ndim == 1:
        out = np.interp(times_target, times_src, vals_src).astype(np.float32)
    else:
        for k in range(vals_src.shape[1]):
            out[:, k] = np.interp(times_target, times_src, vals_src[:, k]).astype(np.float32)
        # Renormalize quaternions
        if vals_src.shape[1] == 4:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms < 1e-6, 1.0, norms)
    return out


def build_layered_glb(base_tracks, overlay_tracks, override_prefixes, out_path: str, target_fps: float | None = None):
    # The output uses the OVERLAY's time domain (length of ESL clip) because that's
    # the actual content. The base idle animation will be sampled into this time domain.
    overlay_times = next(iter(overlay_tracks.values()))[next(iter(next(iter(overlay_tracks.values())).keys()))][0]
    # Pick rotation track for time domain
    for tr in overlay_tracks.values():
        if 'rotation' in tr:
            overlay_times = tr['rotation'][0]; break
    duration = float(overlay_times[-1])
    if target_fps is None:
        # Use overlay's native rate
        target_fps = (len(overlay_times) - 1) / duration if duration > 0 else 25.0
    n_frames = len(overlay_times)
    print(f"Layered output: {n_frames} frames, {duration:.2f}s, ~{target_fps:.1f}fps")

    final_tracks = {}

    # Add overlay-driven bones (override set)
    for bone, paths in overlay_tracks.items():
        if not in_override(bone, override_prefixes):
            continue
        final_tracks[bone] = {}
        for path, (t, v) in paths.items():
            final_tracks[bone][path] = (overlay_times, resample(overlay_times, t, v))

    # Add base-driven bones (everything else)
    for bone, paths in base_tracks.items():
        if in_override(bone, override_prefixes):
            continue
        if bone in final_tracks:
            continue
        final_tracks[bone] = {}
        for path, (t, v) in paths.items():
            # Sample base into the same time domain (loop base if shorter)
            base_duration = float(t[-1]) if len(t) else 0.0
            if base_duration <= 0:
                final_tracks[bone][path] = (overlay_times, np.broadcast_to(v[0:1], (n_frames,) + v.shape[1:]).copy())
                continue
            # Loop base by modulo
            mod_times = overlay_times % base_duration
            sample = resample(mod_times, t, v)
            final_tracks[bone][path] = (overlay_times, sample)

    # Write GLB
    write_anim_glb(final_tracks, overlay_times, out_path)


def write_anim_glb(tracks, times: np.ndarray, out_path: str):
    n_frames = len(times)
    binary = bytearray()

    def append_array(arr):
        nonlocal binary
        if (len(binary) % 4) != 0:
            binary += b'\x00' * (4 - (len(binary) % 4))
        offset = len(binary)
        data = arr.astype(np.float32).tobytes()
        binary += data
        return offset, len(data)

    accessors = []; buffer_views = []

    t_offset, t_len = append_array(times.astype(np.float32))
    buffer_views.append({'buffer': 0, 'byteOffset': t_offset, 'byteLength': t_len})
    accessors.append({'bufferView': 0, 'componentType': 5126, 'count': n_frames,
                      'type': 'SCALAR', 'min': [float(times[0])], 'max': [float(times[-1])]})

    nodes = []; bone_to_idx = {}
    for i, bone in enumerate(tracks.keys()):
        nodes.append({'name': bone}); bone_to_idx[bone] = i
    nodes.append({'name': 'Root', 'children': list(range(len(tracks)))})
    scenes = [{'name': 'Scene', 'nodes': [len(nodes) - 1]}]

    channels = []; samplers = []
    for bone, paths in tracks.items():
        for path, (_, vals) in paths.items():
            type_str = {'translation': 'VEC3', 'rotation': 'VEC4', 'scale': 'VEC3'}[path]
            offset, length = append_array(vals.astype(np.float32))
            bv = len(buffer_views)
            buffer_views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': length})
            acc = len(accessors)
            accessors.append({'bufferView': bv, 'componentType': 5126, 'count': n_frames, 'type': type_str})
            s_idx = len(samplers)
            samplers.append({'input': 0, 'interpolation': 'LINEAR', 'output': acc})
            channels.append({'sampler': s_idx, 'target': {'node': bone_to_idx[bone], 'path': path}})

    animations = [{'name': 'Layered', 'channels': channels, 'samplers': samplers}]

    while len(binary) % 4 != 0:
        binary += b'\x00'

    g = {
        'asset': {'version': '2.0', 'generator': 'layer_animations.py'},
        'scene': 0, 'scenes': scenes, 'nodes': nodes,
        'accessors': accessors, 'bufferViews': buffer_views,
        'buffers': [{'byteLength': len(binary)}],
        'animations': animations,
    }
    json_bytes = json.dumps(g, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out = bytearray()
    out += b'glTF'; out += struct.pack('<I', 2); out += struct.pack('<I', total)
    out += struct.pack('<I', len(json_bytes)); out += b'JSON'; out += json_bytes
    out += struct.pack('<I', len(binary)); out += b'BIN\x00'; out += binary
    Path(out_path).write_bytes(out)
    print(f"✅ Wrote {out_path} ({total} bytes, {len(tracks)} bones, {n_frames} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('overlay')
    ap.add_argument('output_glb')
    ap.add_argument('--override', default=None,
                    help='Comma-separated bone-name prefixes the overlay should override. '
                         'Default = upper body + hands.')
    args = ap.parse_args()

    overrides = tuple(args.override.split(',')) if args.override else DEFAULT_OVERRIDE_PREFIXES

    base = GLTF2().load(args.base)
    overlay = GLTF2().load(args.overlay)
    base_t = get_anim_tracks(base)
    overlay_t = get_anim_tracks(overlay)
    print(f"Base tracks: {len(base_t)} bones | Overlay tracks: {len(overlay_t)} bones")
    print(f"Override prefixes: {overrides}")

    build_layered_glb(base_t, overlay_t, overrides, args.output_glb)


if __name__ == '__main__':
    main()
