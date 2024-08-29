
import numpy as np
import moderngl
import settings
from settings import *


class PaddleManager:
    def __init__(self, ctx):
        # Paddle Properties.
        self.max_paddle_vel = max_paddle_vel

        # LEFT Paddle.
        # Paddle position vertices for the left paddle.
        self.left_paddle_pos = np.array([-0.9, 0.0, -0.88, 0.0, -0.88, 0.4,
                                               -0.9, 0.0, -0.9, 0.4, -0.88, 0.4], dtype='f4')
        # Color values for each vertex of the left paddle.
        self.left_paddle_color = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype='f4')
        # RIGHT Paddle.
        # Paddle position vertices for the right paddle.
        self.right_paddle_pos = np.array([0.88, 0.0, 0.9, 0.0, 0.9, 0.4,
                                                0.88, 0.0, 0.88, 0.4, 0.9, 0.4], dtype='f4')
        # Color values for each vertex of the right paddle.
        self.right_paddle_color = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                                                  0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype='f4')

        # Load Paddle Shaders.
        self.ctx = ctx  # ModernGL context.
        with open('paddle_shader.frag') as f:
            fragment_shader = f.read()  # Load fragment shader.
        with open('paddle_shader.vert') as f:
            vertex_shader = f.read()  # Load vertex shader.
        # Compile shader programs.
        self.paddle_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Initialize VBOs and VAOs for Paddles.
        # VBO and VAO for the left paddle.
        self.left_vbo_paddle_positions = self.ctx.buffer(np.array(self.left_paddle_pos, dtype='f4'))
        self.left_vbo_paddle_colors = self.ctx.buffer(np.array(self.left_paddle_color, dtype='f4'))
        self.left_vao_paddle = self.ctx.vertex_array(
            self.paddle_program,
            [(self.left_vbo_paddle_positions, '2f', 'in_vertex'), (self.left_vbo_paddle_colors, '3f', 'in_color')]
        )
        # VBO and VAO for the right paddle.
        self.right_vbo_paddle_positions = self.ctx.buffer(np.array(self.right_paddle_pos, dtype='f4'))
        self.right_vbo_paddle_colors = self.ctx.buffer(np.array(self.right_paddle_color, dtype='f4'))
        self.right_vao_paddle = self.ctx.vertex_array(
            self.paddle_program,
            [(self.right_vbo_paddle_positions, '2f', 'in_vertex'), (self.right_vbo_paddle_colors, '3f', 'in_color')]
        )

    def update_buffers_then_render_paddles(self):
        """Update VBOs and render paddles."""
        # Update VBO and VAO for the left paddle.
        self.left_vbo_paddle_positions = self.ctx.buffer(np.array(self.left_paddle_pos, dtype='f4'))
        self.left_vbo_paddle_colors = self.ctx.buffer(np.array(self.left_paddle_color, dtype='f4'))
        self.left_vao_paddle = self.ctx.vertex_array(
            self.paddle_program,
            [(self.left_vbo_paddle_positions, '2f', 'in_vertex'), (self.left_vbo_paddle_colors, '3f', 'in_color')]
        )
        # Update VBO and VAO for the right paddle.
        self.right_vbo_paddle_positions = self.ctx.buffer(np.array(self.right_paddle_pos, dtype='f4'))
        self.right_vbo_paddle_colors = self.ctx.buffer(np.array(self.right_paddle_color, dtype='f4'))
        self.right_vao_paddle = self.ctx.vertex_array(
            self.paddle_program,
            [(self.right_vbo_paddle_positions, '2f', 'in_vertex'), (self.right_vbo_paddle_colors, '3f', 'in_color')]
        )
        # Render both paddles.
        self.left_vao_paddle.render(moderngl.TRIANGLES)
        self.right_vao_paddle.render(moderngl.TRIANGLES)

    def check_paddle_hit(self, ball_manager):
        """Check if any ball has hit the paddles and return hit masks and paddle positions."""
        # Left Paddle hit detection.
        mask1 = ball_manager.ball_exists
        mask2 = ball_manager.ball_pos[:, 0] < self.left_paddle_pos[2]
        mask3 = ball_manager.ball_pos[:, 0] > self.left_paddle_pos[0]
        mask4 = ball_manager.ball_pos[:, 1] > self.left_paddle_pos[1]
        mask5 = ball_manager.ball_pos[:, 1] < self.left_paddle_pos[5]
        total_left_mask = mask1 & mask2 & mask3 & mask4 & mask5
        # Right Paddle hit detection.
        mask1 = ball_manager.ball_exists
        mask2 = ball_manager.ball_pos[:, 0] < self.right_paddle_pos[2]
        mask3 = ball_manager.ball_pos[:, 0] > self.right_paddle_pos[0]
        mask4 = ball_manager.ball_pos[:, 1] > self.right_paddle_pos[1]
        mask5 = ball_manager.ball_pos[:, 1] < self.right_paddle_pos[5]
        total_right_mask = mask1 & mask2 & mask3 & mask4 & mask5

        # Return masks and paddle positions.
        return total_left_mask, self.left_paddle_pos[2], total_right_mask, self.right_paddle_pos[0]

    def move_paddles(self, ball_manager):
        """Move paddles based on ball positions and velocities."""
        ball_manager.ball_colors = np.tile(ball_colors, (ball_manager.max_number_of_balls, 1)).astype('f4')
        # LEFT paddle movement.
        mask1 = ball_manager.ball_pos[:, 0] > self.left_paddle_pos[2]  # Balls right of the left paddle.
        mask2 = ball_manager.ball_vel[:, 0] < 0  # Balls moving to the left.
        total_left_mask = mask1 & mask2
        filtered_positions = ball_manager.ball_pos[total_left_mask, 0]
        if filtered_positions.size > 0:
            min_left_index_in_filtered = np.argmin(filtered_positions)
            left_index = np.where(total_left_mask)[0][min_left_index_in_filtered]
        else:
            left_index = None
        if left_index is not None:
            y_direction = 0
            ball_manager.ball_colors[left_index] = np.array([1.0, 0, 0], dtype='f4')
            if ball_manager.ball_pos[left_index, 1] < (self.left_paddle_pos[3] + self.left_paddle_pos[5]) / 2:
                y_direction = -1
            elif ball_manager.ball_pos[left_index, 1] > (self.left_paddle_pos[3] + self.left_paddle_pos[5]) / 2:
                y_direction = 1
            lowest_y_pos = np.min(self.left_paddle_pos[1::2])
            highest_y_pos = np.max(self.left_paddle_pos[1::2])
            if 1 + self.max_paddle_vel > highest_y_pos and lowest_y_pos > -1 - self.max_paddle_vel:
                velocity = np.ones(12, dtype='f4') * self.max_paddle_vel * y_direction
                self.left_paddle_pos[1::2] += velocity[1::2]

        # Prevent the paddle from moving out of the top or bottom bounds.
        if np.min(self.left_paddle_pos[1::2]) <= -1:
            self.left_paddle_pos[1::2] += (np.ones(12, dtype='f4') * 2 * self.max_paddle_vel)[1::2]
        if np.max(self.left_paddle_pos[1::2]) >= 1:
            self.left_paddle_pos[1::2] -= (np.ones(12, dtype='f4') * 2 * self.max_paddle_vel)[1::2]

        # RIGHT paddle movement.
        mask1 = ball_manager.ball_pos[:, 0] < self.right_paddle_pos[0]  # Balls left of the right paddle.
        mask2 = ball_manager.ball_vel[:, 0] > 0  # Balls moving to the right.
        total_right_mask = mask1 & mask2
        filtered_positions = ball_manager.ball_pos[total_right_mask, 0]
        if filtered_positions.size > 0:
            max_right_index_in_filtered = np.argmax(filtered_positions)
            right_index = np.where(total_right_mask)[0][max_right_index_in_filtered]
        else:
            right_index = None
        if right_index is not None:
            ball_manager.ball_colors[right_index] = np.array([0, 1.0, 0], dtype='f4')
            y_direction = 0
            if ball_manager.ball_pos[right_index, 1] < (self.right_paddle_pos[3] + self.right_paddle_pos[5]) / 2:
                y_direction = -1
            elif ball_manager.ball_pos[right_index, 1] > (self.right_paddle_pos[3] + self.right_paddle_pos[5]) / 2:
                y_direction = 1
            lowest_y_pos = np.min(self.right_paddle_pos[1::2])
            highest_y_pos = np.max(self.right_paddle_pos[1::2])
            if 1 + self.max_paddle_vel > highest_y_pos and lowest_y_pos > -1 - self.max_paddle_vel:
                velocity = np.ones(12, dtype='f4') * self.max_paddle_vel * y_direction
                self.right_paddle_pos[1::2] += velocity[1::2]
        # Prevent the paddle from moving out of the top or bottom bounds.
        if np.min(self.right_paddle_pos[1::2]) <= -1:
            self.right_paddle_pos[1::2] += (np.ones(12, dtype='f4') * 2 * self.max_paddle_vel)[1::2]
        if np.max(self.right_paddle_pos[1::2]) >= 1:
            self.right_paddle_pos[1::2] -= (np.ones(12, dtype='f4') * 2 * self.max_paddle_vel)[1::2]


class BallManager:
    def __init__(self, ctx):
        # Initialize Ball Properties.
        self.max_number_of_balls = max_number_of_balls
        self.ball_sizes = np.full(self.max_number_of_balls, ball_size, dtype='f4')
        self.ball_pos = np.ones((self.max_number_of_balls, 2), dtype='f4')
        self.ball_vel = np.ones((self.max_number_of_balls, 2), dtype='f4') * max_ball_vel
        self.ball_exists = np.zeros(self.max_number_of_balls, dtype=bool)
        self.ball_colors = np.tile(ball_colors, (self.max_number_of_balls, 1)).astype('f4')

        # Initialize VBOs and VAO for balls as placeholders.
        self.vbo_ball_positions = np.array([])
        self.vbo_ball_sizes = np.array([])
        self.vbo_ball_colors = np.array([])
        self.vao_ball = np.array([])

        # Load Shaders and Create Program.
        self.ctx = ctx
        with open('ball_shader_03.vert') as f:
            vertex_shader = f.read()
        with open('ball_shader_03.frag') as f:
            fragment_shader = f.read()
        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Ball GLOW Properties, Buffers and Arrays.
        self.ball_glow_sizes = np.full(self.max_number_of_balls, ball_glow_size, dtype='f4')
        self.ball_glow_colors = np.tile(ball_glow_colors, (self.max_number_of_balls, 1)).astype('f4')
        self.vbo_ball_glow_positions = np.array([])
        self.vbo_ball_glow_sizes = np.array([])
        self.vbo_ball_glow_colors = np.array([])
        self.vao_ball_glow = np.array([])

        # Ball Glow Shaders.
        with open('ball_glow_shader_03.vert') as f:
            ball_glow_vertex_shader = f.read()
        with open('ball_glow_shader_03.frag') as f:
            ball_glow_fragment_shader = f.read()
        self.ball_glow_program = self.ctx.program(vertex_shader=ball_glow_vertex_shader,
                                                  fragment_shader=ball_glow_fragment_shader)


    def move_balls(self):
        """Update ball positions based on their velocities."""
        self.ball_pos[self.ball_exists] += self.ball_vel[self.ball_exists]
        # Identify balls out of y bounds.
        y_positions = self.ball_pos[self.ball_exists, 1]
        out_of_y_bounds = (y_positions < -1) | (y_positions > 1)
        # Flip the y velocity for balls out of bounds.
        if np.any(out_of_y_bounds):
            y_indices = np.where(self.ball_exists)[0][out_of_y_bounds]
            self.ball_vel[y_indices, 1] *= -1

    def add_balls(self):
        """Add a new ball to the first empty slot with a certain probability."""
        if np.random.rand() > ball_spawn_chance and np.sum(self.ball_exists) < self.max_number_of_balls:
            # Find the first available slot.
            first_available_slot = np.where(self.ball_exists == False)[0][0]
            self.ball_exists[first_available_slot] = True
            self.ball_pos[first_available_slot] = np.random.uniform(ball_spawn_pos_range[0], ball_spawn_pos_range[1], 2)
            self.ball_vel[first_available_slot] = np.random.uniform(-max_ball_vel, max_ball_vel, 2)

    def destroy_balls(self):
        """Deactivate balls that have gone out of bounds (either left or right)."""
        out_of_x_bounds = (self.ball_pos[self.ball_exists, 0] < -1) | (self.ball_pos[self.ball_exists, 0] > 1)
        out_of_x_bounds_indices = np.where(self.ball_exists)[0][out_of_x_bounds]
        if out_of_x_bounds_indices.size > 0:
            self.ball_exists[out_of_x_bounds_indices] = False
        return out_of_x_bounds_indices

    def update_buffers_then_render_balls(self):
        """Update buffers and render balls."""
        if np.any(self.ball_exists):
            # Create new VBOs with updated data.
            self.vbo_ball_positions = self.ctx.buffer(self.ball_pos[self.ball_exists].tobytes())
            self.vbo_ball_sizes = self.ctx.buffer(self.ball_sizes[self.ball_exists].tobytes())
            self.vbo_ball_colors = self.ctx.buffer(self.ball_colors[self.ball_exists].tobytes())

            # Create or update VAO with the new buffers.
            self.vao_ball = self.ctx.vertex_array(self.program,
                                                  [
                                                      (self.vbo_ball_positions, '2f', 'in_vertex'),
                                                      (self.vbo_ball_colors, '3f', 'in_color'),
                                                      (self.vbo_ball_sizes, '1f', 'in_size')
                                                  ])
            self.vao_ball.render(moderngl.POINTS)

            # For the GLOW Effect.
            self.vbo_ball_glow_positions = self.ctx.buffer(self.ball_pos[self.ball_exists].tobytes())
            self.vbo_ball_glow_sizes = self.ctx.buffer(self.ball_glow_sizes[self.ball_exists].tobytes())
            self.vbo_ball_glow_colors = self.ctx.buffer(self.ball_glow_colors[self.ball_exists].tobytes())
            self.vao_ball_glow = self.ctx.vertex_array(self.ball_glow_program,
                                                       [
                                                           (self.vbo_ball_glow_positions, '2f', 'in_vertex'),
                                                           (self.vbo_ball_glow_colors, '3f', 'in_color'),
                                                           (self.vbo_ball_glow_sizes, '1f', 'in_size')
                                                       ])
            self.vao_ball_glow.render(moderngl.POINTS)

class ParticleManager:
    def __init__(self, ctx):
        # Initialize Particle Properties.
        self.num_particles = num_particles
        self.particle_size = particle_size
        self.max_particle_life = max_particle_life
        self.explosion_centers = []
        self.particle_pos = []
        self.particle_vel = []
        self.particle_colors = []
        self.particle_sizes = []
        self.particle_lives = []

        # Initialize VBOs and VAO for particles as placeholders.
        self.vbo_particle_positions = np.array([])
        self.vbo_particle_colors = np.array([])
        self.vbo_particle_sizes = np.array([])
        self.vao_particle = np.array([])

        # Load Shaders and Create Program.
        self.ctx = ctx
        with open('particle_shader.vert') as f:
            vertex_shader = f.read()
        with open('paddle_shader.frag') as f:
            fragment_shader = f.read()
        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def trigger_explosion(self, positions):
        """Trigger an explosion effect when a ball goes out of bounds."""
        # Define rotation matrix and angles for particle movement.
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_particles)
        rotation_matrices = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]], dtype='f4')
        # Initialize particles for each explosion center.
        for pos in positions:
            self.particle_colors.append(np.tile(np.array(np.random.rand(3)), (self.num_particles, 1)).astype('f4'))
            self.particle_sizes.append(np.ones(self.num_particles, dtype='f4') * self.particle_size)
            velocity = np.array([0.0, 0.0], dtype='f4')
            if pos[0] <= 0:
                self.explosion_centers.append(pos)
                self.particle_lives.append(self.max_particle_life)
                self.particle_pos.append(np.ones((self.num_particles, 2), dtype='f4') * pos)
                velocity = np.array(particle_velocity, dtype='f4')
            elif pos[0] > 0:
                self.explosion_centers.append(pos)
                self.particle_lives.append(self.max_particle_life)
                self.particle_pos.append(np.ones((self.num_particles, 2), dtype='f4') * pos)
                velocity = np.array(particle_velocity, dtype='f4') * -1
            self.particle_vel.append(np.einsum('ijk,j->ik', rotation_matrices, velocity))

    def move_particles(self):
        """Update particle positions and remove those with expired life."""
        indices_to_remove = []
        for counter in range(len(self.explosion_centers)):
            # Update particle positions.
            self.particle_pos[counter] += self.particle_vel[counter].T
            # Decrease particle life.
            self.particle_lives[counter] -= 1
            # Fade out particle color over time.
            self.particle_colors[counter] *= self.particle_lives[counter] / self.max_particle_life
            # Mark particles for removal if their life has ended.
            if self.particle_lives[counter] <= 0:
                indices_to_remove.append(counter)
        # Remove particles after iteration to avoid index errors.
        for index in reversed(indices_to_remove):
            self.particle_lives.pop(index)
            self.explosion_centers.pop(index)
            self.particle_colors.pop(index)
            self.particle_sizes.pop(index)
            self.particle_pos.pop(index)
            self.particle_vel.pop(index)

    def update_buffers_then_render_particles(self):
        """Update buffers and render particles."""
        if len(self.explosion_centers):
            self.vbo_particle_positions = self.ctx.buffer(np.array(self.particle_pos, dtype='f4'))
            self.vbo_particle_colors = self.ctx.buffer(np.array(self.particle_colors, dtype='f4'))
            self.vbo_particle_sizes = self.ctx.buffer(np.array(self.particle_sizes, dtype='f4'))
            self.vao_particle = self.ctx.vertex_array(self.program,
                                                      [
                                                          (self.vbo_particle_positions, '2f', 'in_vertex'),
                                                          (self.vbo_particle_colors, '3f', 'in_color'),
                                                          (self.vbo_particle_sizes, '1f', 'in_size')
                                                      ])
            self.vao_particle.render(moderngl.POINTS)