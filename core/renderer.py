import ctypes
import numpy as np
from OpenGL import GL as gl


class InstancedBoxRenderer:
    MAX_INSTANCES = 8192

    def __init__(self, shader):
        self.shader = shader
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vertices, normals = self._make_unit_cube()
        geom_data = np.hstack([vertices, normals]).astype(np.float32)

        self.vbo_geom = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_geom)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, geom_data.nbytes, geom_data, gl.GL_STATIC_DRAW)

        stride = 6 * 4  # 6 floats * 4 bytes
        # position
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        # normal
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))

        # Instance buffer: pos(3) + color(4) + scale(3) = 10 floats = 40 bytes
        self.vbo_instance = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_instance)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.MAX_INSTANCES * 10 * 4,
                        None, gl.GL_DYNAMIC_DRAW)

        inst_stride = 10 * 4  # 40 bytes
        # instance position
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, inst_stride, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(2, 1)
        # instance color
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, inst_stride, ctypes.c_void_p(12))
        gl.glVertexAttribDivisor(3, 1)
        # instance scale
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, inst_stride, ctypes.c_void_p(28))
        gl.glVertexAttribDivisor(4, 1)

        gl.glBindVertexArray(0)

    def destroy(self):
        gl.glDeleteBuffers(2, [self.vbo_geom, self.vbo_instance])
        gl.glDeleteVertexArrays(1, [self.vao])

    def draw(self, instance_data: np.ndarray, count: int):
        if count <= 0:
            return
        self.shader.use()
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_instance)
        data = np.ascontiguousarray(instance_data[:count], dtype=np.float32)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, count)
        gl.glBindVertexArray(0)

    def _make_unit_cube(self):
        # Centered at origin, size 1x1x1 (-0.5 to 0.5)
        v = np.array([
            # Front face (z = +0.5)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
            [-0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            # Back face (z = -0.5)
            [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5],
            [ 0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            # Top face (y = +0.5)
            [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
            [-0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            # Bottom face (y = -0.5)
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
            [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # Right face (x = +0.5)
            [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5],
            [ 0.5, -0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
            # Left face (x = -0.5)
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        ], dtype=np.float32)

        n = np.array([
            # Front
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Back
            [0, 0, -1], [0, 0, -1], [0, 0, -1],
            [0, 0, -1], [0, 0, -1], [0, 0, -1],
            # Top
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            [0, 1, 0], [0, 1, 0], [0, 1, 0],
            # Bottom
            [0, -1, 0], [0, -1, 0], [0, -1, 0],
            [0, -1, 0], [0, -1, 0], [0, -1, 0],
            # Right
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            [1, 0, 0], [1, 0, 0], [1, 0, 0],
            # Left
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
        ], dtype=np.float32)

        return v, n
