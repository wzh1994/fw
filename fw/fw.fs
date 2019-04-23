#version 400
in vec3 ocolor;
out vec4 frag_colour;
void main() {
	frag_colour = vec4(ocolor, 1.0);
}