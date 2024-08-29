# PONG-Autonomous_Pong_Simulation
ModernGL Python code for self playing PONG game. 

## Overview
PONG: Autonomous Pong Simulation is designed to be completely autonomous. Utilizing the power of OpenGL through the ModernGL library in Python, this project showcases how basic shaders and rendering techniques can be leveraged for simple yet visually appealing simulations.

Project mainly acts as my cheat sheet for various moderngl and numpy methods.


https://github.com/user-attachments/assets/b16d0258-fbdd-44f0-ab1b-37a0e3821bec


## Features
Self-Playing: Both paddles are controlled by the logic, eliminating the need for user interaction.

Glowing Ball Effect: The ball features a subtle glow effect using fragment shaders, giving it a modern, polished look.

Efficient Rendering: Leveraging ModernGL for fast, GPU-based rendering.

Easy Configuration: Centralized settings for easy adjustments and fine-tuning of game properties.

## Getting Started
### Prerequisites
- Only tested in Python 3.12, so Python 3.12.
- moderngl
- moderngl-window
- numpy

Install the required dependencies

```pip install moderngl moderngl-window numpy```

### Running the Simulation
Run Pong_03.py directly. 

```python Pong_03.py```

### Customization
Most properties, such as paddle speed, ball speed, colors and sizes are configurable in the settings.py file. 
Shaders may have a few magic numbers that may need modifications.

## Code Structure
Pong_03.py: The main entry point. It needs to be run directly.
Shaders: There are a few shaders that control glow or transformations. Shaders will be listed with more details below.
Pong_Objects_03.py: Defines the ball, paddle and explosion particles as well as their behavior.

## Shaders
- paddle_shader.vert: Handles paddle vertices.
- paddle_shader.frag: Defines paddle colors.
- ball_shader_03.vert: Handles ball vertices.
- ball_shader_03.frag: Defines ball colors.
- ball_glow_shader_03.vert: Handles the vertices of the glowing effect on balls.
- ball_glow_shader_03.frag: Defines the colors for the flowing effect on balls.
- particle_shader.vert: Handles the vertices of the particles created upon explosions of the balls.
- particle_shader.frag: Defines the colors of the particles created upon explosions of the balls.

## Future Enhancements
- Separating glow shaders and logic from BallManager class.
- Visual Enhancements: More glows, even for explosion particles.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
Or you know, go wild.





















