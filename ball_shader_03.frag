#version 330
in vec3 v_color;
out vec4 fragColor;

void main(){
    vec2 center = vec2(0.5, 0.5); // Center of the point in normalized coordinates
    float dist = length(gl_PointCoord - center); // Distance from the center

    if (dist > 0.5) {
                discard;  // Discard fragments outside the circle
    }
    fragColor = vec4(v_color, 1);
}