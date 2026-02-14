#version 410 core

in vec3 v_normal;
in vec4 v_color;
in vec3 v_frag_pos;

out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(-u_light_dir);

    float ambient = 0.25;
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(u_camera_pos - v_frag_pos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3;

    vec3 lit_color = v_color.rgb * (ambient + diff * 0.65) + vec3(spec);
    frag_color = vec4(lit_color, v_color.a);
}
