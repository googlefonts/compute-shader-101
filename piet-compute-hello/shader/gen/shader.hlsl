static const uint3 gl_WorkGroupSize = uint3(128u, 1u, 1u);

RWByteAddressBuffer _10 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    _10.Store(gl_GlobalInvocationID.x * 4 + 0, _10.Load(gl_GlobalInvocationID.x * 4 + 0) + 42u);
}

[numthreads(128, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
