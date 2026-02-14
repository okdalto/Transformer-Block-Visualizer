import numpy as np
from OpenGL import GL as gl


class ShaderProgram:
    def __init__(self, vert_path: str, frag_path: str):
        vert_src = self._read_file(vert_path)
        frag_src = self._read_file(frag_path)

        vert_shader = self._compile_shader(vert_src, gl.GL_VERTEX_SHADER)
        frag_shader = self._compile_shader(frag_src, gl.GL_FRAGMENT_SHADER)

        self.program_id = gl.glCreateProgram()
        gl.glAttachShader(self.program_id, vert_shader)
        gl.glAttachShader(self.program_id, frag_shader)
        gl.glLinkProgram(self.program_id)

        if gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            info = gl.glGetProgramInfoLog(self.program_id).decode()
            raise RuntimeError(f"Shader link error:\n{info}")

        gl.glDeleteShader(vert_shader)
        gl.glDeleteShader(frag_shader)

        self._uniform_cache = {}

    def _read_file(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    def _compile_shader(self, source: str, shader_type) -> int:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            info = gl.glGetShaderInfoLog(shader).decode()
            type_name = "VERTEX" if shader_type == gl.GL_VERTEX_SHADER else "FRAGMENT"
            raise RuntimeError(f"{type_name} shader compile error:\n{info}")
        return shader

    def destroy(self):
        gl.glDeleteProgram(self.program_id)

    def use(self):
        gl.glUseProgram(self.program_id)

    def _get_loc(self, name: str) -> int:
        if name not in self._uniform_cache:
            self._uniform_cache[name] = gl.glGetUniformLocation(self.program_id, name)
        return self._uniform_cache[name]

    def set_mat4(self, name: str, matrix: np.ndarray):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, matrix.astype(np.float32))

    def set_vec3(self, name: str, v: np.ndarray):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform3fv(loc, 1, v.astype(np.float32))

    def set_float(self, name: str, value: float):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform1f(loc, value)

    def set_int(self, name: str, value: int):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform1i(loc, value)

    def set_vec4(self, name: str, v: np.ndarray):
        loc = self._get_loc(name)
        if loc >= 0:
            gl.glUniform4fv(loc, 1, v.astype(np.float32))
