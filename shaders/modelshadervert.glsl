#version 320 es

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fPos;
out vec2 fTexCoord;
out vec3 fNormal;

void main()
{
    fNormal = mat3(transpose(inverse(model))) * vNormal;
    fTexCoord = vTexCoord;
	fPos = vec3(model * vec4(vPos, 1.0));
    gl_Position = projection * view * model * vec4(vPos, 1.f);
}
