#version 410 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 2) in vec3 a_instance_pos;
layout(location = 3) in vec4 a_instance_color;
layout(location = 4) in vec3 a_instance_scale;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec4 v_color;
out vec3 v_frag_pos;

void main() {
    vec3 world_pos = a_position * a_instance_scale + a_instance_pos;
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
    v_normal = normalize(a_normal / a_instance_scale);
    v_color = a_instance_color;
    v_frag_pos = world_pos;
}
