#version 410 core

in vec2 v_texcoord;

uniform sampler2D u_font_atlas;
uniform vec4 u_text_color;
uniform float u_depth;  // >= 0: override fragment depth, < 0: no override

out vec4 frag_color;

void main() {
    float alpha = texture(u_font_atlas, v_texcoord).r;
    frag_color = vec4(u_text_color.rgb, u_text_color.a * alpha);
    // Write custom depth for 3D-projected labels; pass through for 2D overlay
    gl_FragDepth = (u_depth >= 0.0) ? u_depth : gl_FragCoord.z;
}
