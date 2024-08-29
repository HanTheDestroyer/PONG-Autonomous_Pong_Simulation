#version 330
out vec4 fragColor;
in vec3 v_color;

void main() {
    vec2 center = vec2(0.5, 0.5); // Center of the point in normalized coordinates
    float dist = length(gl_PointCoord - center); // Distance from the center
    float pointRadius = 0.08;  // Radius of the solid point
    float glowRadius = 0.5;   // Large radius for the glow
    vec3 glowColor = v_color; // Fixed orange glow color
    if (dist < glowRadius) {
        float alpha = 0.0;
        if (dist < pointRadius) {
            alpha = 1.0;  // Fully opaque inside the point
        } else {
            // Smooth fade for the glow
            alpha = pow(1.0 - (dist - pointRadius) / (glowRadius - pointRadius), 2.0);
        }
        fragColor = vec4(glowColor, alpha);
    } else {
        discard;
    }
}
