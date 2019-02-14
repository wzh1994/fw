#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in mat4 models;
layout (location = 5) in vec3 color;
out vec3 itemColor;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * models * vec4(position, 1.0f);
	itemColor = color;
}