
#version 330
in vec2 in_vertex;
in vec3 in_color;
out vec3 v_color;
void main(){
    v_color = in_color;
    gl_Position = vec4(in_vertex, 0.0, 1.0);
}
