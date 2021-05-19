struct VertexOutput {
    [[location(0)]] tex_coord: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] in_vertex_index: u32,
    [[builtin(instance_index)]] in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((in_vertex_index & 1u) ^ in_instance_index);
    let y = f32((in_vertex_index >> 1u) ^ in_instance_index);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.tex_coord = vec2<f32>(x, y);
    return out;
}

[[group(0), binding(0)]]
var r_color: texture_2d<f32>;
[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_color, r_sampler, in.tex_coord);
}
