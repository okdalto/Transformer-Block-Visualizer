import os
import sys

os.environ['GL_SILENCE_DEPRECATION'] = '1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glfw
import numpy as np
from OpenGL import GL as gl

from core.window import AppWindow
from core.shader import ShaderProgram
from core.text_renderer import TextRenderer
from transformer.parameters import TransformerConfig
from transformer.computation import TransformerBlock
from visualization.scene import Scene


def main():
    app = AppWindow(width=1600, height=900, title="Transformer Block Visualizer")

    # Shader paths relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vert_path = os.path.join(base_dir, "shaders", "box_instanced.vert")
    frag_path = os.path.join(base_dir, "shaders", "box_instanced.frag")
    box_shader = ShaderProgram(vert_path, frag_path)

    text_vert = os.path.join(base_dir, "shaders", "text.vert")
    text_frag = os.path.join(base_dir, "shaders", "text.frag")
    text_renderer = TextRenderer(text_vert, text_frag, font_size=42)
    label_renderer = TextRenderer(text_vert, text_frag, font_size=27)

    # Run transformer computation
    config = TransformerConfig()
    transformer = TransformerBlock(config)
    x = np.random.RandomState(123).randn(config.seq_len, config.d_model).astype(np.float32) * 0.5
    results = transformer.forward(x)

    # Build scene
    scene = Scene(results, config, box_shader)

    # Keyboard handler
    def on_key(key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(app.window, True)
        elif key == glfw.KEY_SPACE:
            scene.timeline.toggle_play()
        elif key == glfw.KEY_RIGHT:
            idx = scene.timeline.get_current_stage_index()
            scene.timeline.jump_to_stage(idx + 1)
        elif key == glfw.KEY_LEFT:
            idx = scene.timeline.get_current_stage_index()
            scene.timeline.jump_to_stage(idx - 1)
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            scene.timeline.speed = min(5.0, scene.timeline.speed + 0.25)
            print(f"Speed: {scene.timeline.speed:.2f}x")
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            scene.timeline.speed = max(0.25, scene.timeline.speed - 0.25)
            print(f"Speed: {scene.timeline.speed:.2f}x")
        elif key == glfw.KEY_R:
            scene.timeline.current_time = 0.0
            scene.timeline.playing = True
            print("Reset to beginning")

    def on_mouse(dx, dy):
        scene.camera.handle_mouse(dx, dy)

    def on_scroll(xoff, yoff):
        scene.camera.handle_scroll(xoff, yoff)

    app.add_key_callback(on_key)
    app.add_mouse_callback(on_mouse)
    app.add_scroll_callback(on_scroll)

    # OpenGL state
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_MULTISAMPLE)
    gl.glClearColor(0.08, 0.08, 0.12, 1.0)

    last_time = glfw.get_time()
    last_stage = ""

    print("=== Transformer Block Visualizer ===")
    print("Controls:")
    print("  Space      : Play / Pause")
    print("  Left/Right : Previous / Next stage")
    print("  +/-        : Speed up / Slow down")
    print("  R          : Reset to beginning")
    print("  Mouse drag : Orbit camera")
    print("  Scroll     : Zoom")
    print("  Esc        : Quit")
    print()
    sys.stdout.flush()

    while not app.should_close():
        current_time = glfw.get_time()
        dt = min(current_time - last_time, 0.05)
        last_time = current_time

        scene.update(dt)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        fb_w, fb_h = app.get_framebuffer_size()
        aspect = fb_w / max(fb_h, 1)
        scene.render(aspect)

        # 3D projected labels
        scene.render_labels(label_renderer, fb_w, fb_h)

        # Text overlay
        stage_idx = scene.timeline.get_current_stage_index()
        stage = scene.timeline.stages[stage_idx]
        text_renderer.render_stage_name(stage.stage_name, fb_w, fb_h)

        # Print stage transitions
        phase, phase_t = stage.get_phase(scene.timeline.current_time)
        stage_id = f"{stage.stage_name}:{phase}"
        if stage_id != last_stage:
            last_stage = stage_id
            print(f"  [{stage.stage_name}] {phase} (t={scene.timeline.current_time:.1f}s)")
            sys.stdout.flush()

        app.swap_and_poll()

    scene.renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()


if __name__ == '__main__':
    main()
