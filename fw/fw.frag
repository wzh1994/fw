#version 330 core
out vec4 color;
in vec3 itemColor;

void main()
{
    color =vec4(itemColor,1.0f) ;
}