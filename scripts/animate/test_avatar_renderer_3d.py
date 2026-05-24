"""Quick test: render a 2-token sequence via the new platform service."""
import asyncio, sys
sys.path.insert(0, '/root/.openclaw/workspace/esl-platform/backend')

# Provide a minimal settings stub if needed
import os
os.environ.setdefault('PYTHONPATH', '/root/.openclaw/workspace/esl-platform/backend')

async def main():
    from pathlib import Path
    from app.services.avatar_renderer_3d import Avatar3DRenderer, Render3DConfig
    cfg = Render3DConfig(
        width=600, height=700, fps=25,
        output_dir=Path('/root/.openclaw/workspace/test_3d_renders'),
    )
    r = Avatar3DRenderer(cfg)
    result = await r.render_sequence(['DOCTOR', 'PLAYS', 'FAMILY'], output_name='test_multi')
    print(f"✓ Rendered: {result.output_path}")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Size: {result.file_size_bytes / 1024:.1f} KB")
    print(f"  Tokens rendered: {result.tokens_rendered}")
    print(f"  Tokens missing: {result.tokens_missing}")

asyncio.run(main())
