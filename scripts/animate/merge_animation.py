#!/usr/bin/env python3
"""
Merge a Mixamo-style animation GLB into a Ready Player Me (or any Mixamo-rigged) avatar GLB.

The animation's per-bone tracks are remapped from the animation file's node indices
to the avatar's node indices by matching bone names.

Usage:
  python merge_animation.py <avatar.glb> <animation.glb> <output.glb>
"""

import sys
import json
import struct
from pathlib import Path
from pygltflib import GLTF2, Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler, Accessor, BufferView, Buffer


def merge(avatar_path: str, anim_path: str, out_path: str):
    print(f"Loading avatar: {avatar_path}")
    avatar = GLTF2().load(avatar_path)
    print(f"  nodes: {len(avatar.nodes)}, animations: {len(avatar.animations)}, accessors: {len(avatar.accessors)}, bufferViews: {len(avatar.bufferViews)}")

    print(f"Loading animation: {anim_path}")
    anim = GLTF2().load(anim_path)
    print(f"  nodes: {len(anim.nodes)}, animations: {len(anim.animations)}")

    # Build name -> avatar_node_index map
    # Support both plain ("Hips") and prefixed ("mixamorig9:Hips") bone names.
    def _strip(name):
        for pfx in ('mixamorig9:', 'mixamorig:', 'mixamorig1:'):
            if name.startswith(pfx): return name[len(pfx):]
        return name
    avatar_name_to_idx = {n.name: i for i, n in enumerate(avatar.nodes) if n.name}
    avatar_stripped_to_idx = {_strip(n.name): i for i, n in enumerate(avatar.nodes) if n.name}

    # Build animation_node_index -> name map
    anim_idx_to_name = {i: n.name for i, n in enumerate(anim.nodes) if n.name}

    # --- Extract avatar's binary buffer ---
    if not avatar.buffers or avatar.buffers[0].uri is not None:
        # For .glb, the buffer is embedded; pygltflib handles this via binary_blob()
        pass
    avatar_bin = avatar.binary_blob()
    anim_bin = anim.binary_blob()
    print(f"  avatar binary: {len(avatar_bin)} bytes")
    print(f"  animation binary: {len(anim_bin)} bytes")

    # --- Append animation binary to avatar binary (aligned to 4 bytes) ---
    align_pad = (4 - (len(avatar_bin) % 4)) % 4
    new_bin = avatar_bin + (b"\x00" * align_pad) + anim_bin
    anim_bin_offset_in_new = len(avatar_bin) + align_pad

    # --- Copy animation bufferViews into avatar, offsetting byteOffset by anim_bin_offset_in_new ---
    base_buffer_view_index = len(avatar.bufferViews)
    for bv in anim.bufferViews:
        new_bv = BufferView()
        new_bv.buffer = 0  # avatar always has a single buffer
        new_bv.byteOffset = (bv.byteOffset or 0) + anim_bin_offset_in_new
        new_bv.byteLength = bv.byteLength
        new_bv.byteStride = bv.byteStride
        new_bv.target = bv.target
        avatar.bufferViews.append(new_bv)

    # --- Copy animation accessors, remapping bufferView index ---
    base_accessor_index = len(avatar.accessors)
    for acc in anim.accessors:
        new_acc = Accessor()
        new_acc.bufferView = (acc.bufferView or 0) + base_buffer_view_index
        new_acc.byteOffset = acc.byteOffset
        new_acc.componentType = acc.componentType
        new_acc.normalized = acc.normalized
        new_acc.count = acc.count
        new_acc.type = acc.type
        new_acc.max = acc.max
        new_acc.min = acc.min
        new_acc.sparse = acc.sparse
        avatar.accessors.append(new_acc)

    # --- Update avatar buffer byteLength ---
    avatar.buffers[0].byteLength = len(new_bin)
    avatar.set_binary_blob(new_bin)

    # --- Convert animations ---
    matched, unmatched = 0, []
    for src_anim in anim.animations:
        new_anim = Animation()
        new_anim.name = src_anim.name
        new_anim.samplers = []
        new_anim.channels = []

        # Copy samplers, remapping accessor indices
        for s in src_anim.samplers:
            new_s = AnimationSampler()
            new_s.input = (s.input or 0) + base_accessor_index
            new_s.output = (s.output or 0) + base_accessor_index
            new_s.interpolation = s.interpolation
            new_anim.samplers.append(new_s)

        # Copy channels, remapping target.node via name lookup
        for ch in src_anim.channels:
            src_node_idx = ch.target.node
            bone_name = anim_idx_to_name.get(src_node_idx)
            if bone_name is None:
                unmatched.append(f"<no-name idx={src_node_idx}>")
                continue
            if bone_name not in avatar_name_to_idx:
                stripped = _strip(bone_name)
                if stripped in avatar_stripped_to_idx:
                    resolved_idx = avatar_stripped_to_idx[stripped]
                    new_ch = AnimationChannel()
                    new_ch.sampler = ch.sampler
                    new_target = AnimationChannelTarget()
                    new_target.node = resolved_idx
                    new_target.path = ch.target.path
                    new_ch.target = new_target
                    new_anim.channels.append(new_ch)
                    matched += 1
                    continue
                unmatched.append(bone_name)
                continue
            new_ch = AnimationChannel()
            new_ch.sampler = ch.sampler
            new_target = AnimationChannelTarget()
            new_target.node = avatar_name_to_idx[bone_name]
            new_target.path = ch.target.path
            new_ch.target = new_target
            new_anim.channels.append(new_ch)
            matched += 1

        avatar.animations.append(new_anim)

    print(f"  Matched {matched} animation channels to avatar bones.")
    if unmatched:
        print(f"  ⚠️ Unmatched bones (skipped): {sorted(set(unmatched))}")

    # --- Save ---
    avatar.save(out_path)
    print(f"✅ Wrote {out_path}")
    print(f"   Total nodes: {len(avatar.nodes)}")
    print(f"   Total animations: {len(avatar.animations)}")
    print(f"   Total accessors: {len(avatar.accessors)}")
    print(f"   Total bufferViews: {len(avatar.bufferViews)}")
    print(f"   Buffer size: {len(new_bin)} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
