#version 450

layout(push_constant) uniform VertexConstants {
    vec2 u_screen_size;
} pc;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_tex_coord;
layout(location = 2) in uint a_color;
layout(location = 0) out vec2 v_tex_coord;
layout(location = 1) out vec4 v_color;

void main() {
    v_tex_coord = a_tex_coord;
    // [u8; 4] SRGB as u32 -> [r, g, b, a]
    v_color = vec4(a_color & 0xFFu, (a_color >> 8) & 0xFFu, (a_color >> 16) & 0xFFu, (a_color >> 24) & 0xFFu) / 255.0;
    gl_Position = vec4(2.0 * a_pos.x / pc.u_screen_size.x - 1.0, 1.0 - 2.0 * a_pos.y / pc.u_screen_size.y, 0.0, 1.0);
}
