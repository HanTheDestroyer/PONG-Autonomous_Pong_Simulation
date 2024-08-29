import moderngl
from moderngl_window import WindowConfig
import numpy as np
from Project_03_Pong.Pong_Objects_03 import BallManager, ParticleManager, PaddleManager

class Animation(WindowConfig):
    # Set OpenGL version and window properties.
    gl_version = (3, 3)
    vsync = True
    resizable = True
    aspect_ratio = None
    window_size = (1280, 720)
    title = 'Exploding Balls'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Enable point size control in ModernGL.
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        # Initialize managers for ball, particle, and paddle.
        self.ball_manager = BallManager(self.ctx)
        self.particle_manager = ParticleManager(self.ctx)
        self.paddle_manager = PaddleManager(self.ctx)

    def update_buffers(self):
        """Update VBOs with current data from ball, paddle, and particle managers."""
        self.paddle_manager.update_buffers_then_render_paddles()
        self.ball_manager.update_buffers_then_render_balls()
        self.particle_manager.update_buffers_then_render_particles()

    def render(self, time: float, frame_time: float):
        """Update and render the animation. Handle ball movements and collisions."""
        # Add new balls if conditions are met.
        self.ball_manager.add_balls()
        self.ctx.clear()  # Clear the screen for new frame rendering.
        # Move particles and paddles.
        self.particle_manager.move_particles()
        self.paddle_manager.move_paddles(self.ball_manager)
        # Check for paddle collisions with balls.
        left_mask, left_paddle_pos, right_mask, right_paddle_pos = self.paddle_manager.check_paddle_hit(self.ball_manager)
        # Update ball velocities and positions based on paddle hits.
        if np.any(left_mask):
            self.ball_manager.ball_vel[left_mask] *= -1
            self.ball_manager.ball_pos[left_mask, 0] = left_paddle_pos
        if np.any(right_mask):
            self.ball_manager.ball_vel[right_mask] *= -1
            self.ball_manager.ball_pos[right_mask, 0] = right_paddle_pos
        # Move balls and update buffers.
        self.ball_manager.move_balls()
        self.update_buffers()
        # Destroy balls that are out of bounds and trigger particle explosions.
        indices = self.ball_manager.destroy_balls()
        centers = self.ball_manager.ball_pos[indices]
        if centers.size > 0:
            self.particle_manager.trigger_explosion(centers)

if __name__ == '__main__':
    Animation.run()
