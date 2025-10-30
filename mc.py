import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from astropy import constants as const

# constants
mean_free_path = 150.0  # meters
sigma_T = 6.652e-25  # thomson cross section [cm^2]
slab_width = 1000.0  # meters (1 km)


# initializing the photon's state
start_x = slab_width / 2

start_y = slab_width / 2

pos_x_history = [start_x]

pos_y_history = [start_y]

pos_z_history = [slab_width / 2]

total_path_length = 0.0

photon_escaped = False

# parameters
max_scatters = 15
animation_delay = 100  # milliseconds

def get_scatter_distance(mfp=mean_free_path):
    """
    draws random distance from exponential distribution.
    this uses inverse transform sampling: if P ~ Uniform(0,1), 
    then -L*ln(P) ~ exponential(1/L)
    """
    return -mfp * np.log(np.random.rand())

def get_scatter_angle(): 
    """
    for isotropic scattering in 2D, angle is uniform on (0, 2π)
    this returns the angle in radians
    """
    return 2 * np.pi * np.random.rand()

def perform_scatter():
    """
    this simulated one scattering event and
    updates photon position to checks if it has exited the slab
    """
    global pos_x_history, pos_y_history, pos_z_history, total_path_length, photon_escaped
    
    if photon_escaped:
        return
    
    # current position
    current_x = pos_x_history[-1]
    current_y = pos_y_history[-1]
    current_z = pos_z_history[-1]
    
    # get distance to next scatter and angle
    step_length = get_scatter_distance()
    azimuthal = get_scatter_angle()
    polar = np.arccos(2 * np.random.rand() - 1)  # 3D isotropic
    
    # calculate new position
    dx = step_length * np.sin(polar) * np.cos(azimuthal)
    dy = step_length * np.sin(polar) * np.sin(azimuthal)
    dz = step_length * np.cos(polar)
    
    next_x = current_x + dx
    next_y = current_y + dy
    next_z = current_z + dz
    
    """updating the arrays"""
    pos_x_history.append(next_x)
    pos_y_history.append(next_y)
    pos_z_history.append(next_z)
    total_path_length += step_length
    
    """checking if the photon exited slab boundaries"""
    if next_x < 0 or next_x > slab_width or next_y < 0 or next_y > slab_width or next_z < 0 or next_z > slab_width:
        photon_escaped = True

"""setting up the figure and axis"""
canvas = plt.figure(figsize=(11, 9))
plot_ax = canvas.add_subplot(111, projection='3d')
plot_ax.set_xlim(-slab_width*0.2, slab_width*1.2)
plot_ax.set_ylim(-slab_width*0.2, slab_width*1.2)
plot_ax.set_zlim(-slab_width*0.2, slab_width*1.2)
plot_ax.set_xlabel('x (m)', fontsize=12)
plot_ax.set_ylabel('y (m)', fontsize=12)
plot_ax.set_zlabel('z (m)', fontsize=12)

# draw wireframe cube for slab
cube_vertices = [
    [0, 0, 0], [slab_width, 0, 0], [slab_width, slab_width, 0], [0, slab_width, 0],
    [0, 0, slab_width], [slab_width, 0, slab_width], [slab_width, slab_width, slab_width], [0, slab_width, slab_width]
]
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
    [4, 5], [5, 6], [6, 7], [7, 4],  # top
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
]
for edge in edges:
    pts = [cube_vertices[edge[0]], cube_vertices[edge[1]]]
    plot_ax.plot3D(*zip(*pts), color='darkblue', linestyle='--', linewidth=2, alpha=0.6)

# draw transparent portion
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
faces = [
    [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],
    [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]],
    [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],
    [cube_vertices[3], cube_vertices[0], cube_vertices[4], cube_vertices[7]],
    [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]],
    [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]]
]
cube_collection = Poly3DCollection(faces, alpha=0.08, facecolor='cyan', edgecolor='none')
plot_ax.add_collection3d(cube_collection)

"""creating plot elements"""
trajectory_line, = plot_ax.plot([], [], [], 'purple', linewidth=2, alpha=0.8, label='Photon Path')
collision_markers, = plot_ax.plot([], [], [], 'o', color='orange', markersize=6, alpha=0.7, label='Scatter Events')
active_position, = plot_ax.plot([], [], [], 'o', color='lime', markersize=12, label='Current Position')

# text box that will show the stats of travel
stat_display = plot_ax.text2D(0.02, 0.98, '', transform=plot_ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plot_ax.legend(loc='upper right', fontsize=10)
# in order to get a realistic estimate we need to run many trials, i did 1000
n_trials = 1000
escaped = 0
reflected = 0

for _ in range(n_trials):
    x, y = slab_width/2, slab_width/2
    while 0 <= x <= slab_width and 0 <= y <= slab_width:
        dist = get_scatter_distance()
        theta = get_scatter_angle()
        x += dist * np.cos(theta)
        y += dist * np.sin(theta)
    
    if x > slab_width or y > slab_width:
        escaped += 1
    else:
        reflected += 1

print(f"Results from {n_trials} photons:")
print(f"Escape fraction: {escaped/n_trials:.3f}")
print(f"Reflection fraction: {reflected/n_trials:.3f}")
print()

def init_animation():
    """this initializes the animation"""
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    collision_markers.set_data([], [])
    collision_markers.set_3d_properties([])
    active_position.set_data([], [])
    active_position.set_3d_properties([])
    stat_display.set_text('')
    return trajectory_line, collision_markers, active_position, stat_display

def animate_frame(frame):
    """this updates the animation for each frame"""
    global pos_x_history, pos_y_history, pos_z_history, total_path_length, photon_escaped
    
    # this performs another scatter if the photon has not yet exited
    if not photon_escaped:
        perform_scatter()
    
    # this updates the line path line
    trajectory_line.set_data(pos_x_history, pos_y_history)
    trajectory_line.set_3d_properties(pos_z_history)
    
    # this updates scatter points (all but current position)
    if len(pos_x_history) > 1:
        collision_markers.set_data(pos_x_history[:-1], pos_y_history[:-1])
        collision_markers.set_3d_properties(pos_z_history[:-1])
    
    # this updates the current position
    active_position.set_data([pos_x_history[-1]], [pos_y_history[-1]])
    active_position.set_3d_properties([pos_z_history[-1]])
    
    # calculates the time, which is distance divided by speed of light 
    time_microsec = (total_path_length / 3e8) * 1e6  # in microseconds
    
    # title
    status = "Exited!" if photon_escaped else "Scattering in Progress"
    plot_ax.set_title(f'Photon Scattering in 1km Slab | Scatters: {len(pos_x_history)-1} | {status}',
                 fontsize=13, fontweight='bold')
    
    # updating info box
    info_text = f'Distance: {total_path_length:.2f} m\n'
    info_text += f'Time: {time_microsec:.3f} μs\n'
    info_text += f'Scatters: {len(pos_x_history)-1}'
    stat_display.set_text(info_text)
    
    # rotates the 3d box in order to show a cool visual!
    plot_ax.view_init(elev=20, azim=frame * 2)
    
    return trajectory_line, collision_markers, active_position, stat_display

"""creating the animation!"""
slab_anim = animation.FuncAnimation(
    canvas, 
    animate_frame,
    init_func=init_animation,
    frames=max_scatters,
    interval=animation_delay,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()


sigma_T = 6.652e-25  # Thomson cross section [cm^2]
R_sun_m = const.R_sun.to(u.m).value  # Solar radius in meters
solar_max_scatters = 100
solar_time_delay = 150

# visualization scale factor, since the real MFP is too small to see
scale_factor = 1e11

# initialize solar photon at center
solar_start_x = 0.0
solar_start_y = 0.0
solar_start_z = 0.0
solar_x_positions = [solar_start_x]
solar_y_positions = [solar_start_y]
solar_z_positions = [solar_start_z]
solar_total_dist = 0.0
solar_sim_exit = False
solar_time = 0.0 * u.s

def compute_mfp_at_radius(r):
    """
    calculating the mean free path based on radial position in the Sun.
    this uses exponential the electron density profile
    
    Parameters:
        r: distance from Sun's center (meters)
    
    Returns:
        Mean free path in meters (scaled for visualization)
    """
    # electron density formula: n_e(r) = 2.5×10^26 * exp(-r / (0.096 R_sun))
    scale_height = 0.096 * R_sun_m
    n_e_base = 2.5e26  # cm^-3
    
    electron_density = n_e_base * np.exp(-r / scale_height)  # in cm^-3
    
    # Mean free path: l = 1 / (n_e * sigma_T)
    mfp_cm = 1.0 / (electron_density * sigma_T)
    mfp_m = mfp_cm * 1e-2  # convert to meters
    
    return mfp_m * scale_factor

def scatter_photon_3d():
    """
    Perform one 3D isotropic scattering event in solar interior.
    Updates global position arrays and distance tracker.
    
    Returns:
        current_r: distance from Sun's center after scatter
        has_escaped: boolean indicating if photon left the Sun
    """
    global solar_x_positions, solar_y_positions, solar_z_positions
    global solar_total_dist, solar_sim_exit, solar_time
    
    if solar_sim_exit:
        return 0.0, True
    
    # obtaining the current positions
    x_curr = solar_x_positions[-1]
    y_curr = solar_y_positions[-1]
    z_curr = solar_z_positions[-1]
    
    # calculating radius from center
    r_curr = np.sqrt(x_curr**2 + y_curr**2 + z_curr**2)
    
    # calculating mean free path from center
    step_distance = -compute_mfp_at_radius(r_curr) * np.log(np.random.rand())
    
    # isotropic 3D scattering angles
    # theta: polar angle from z-axis [0, π]
    # phi: azimuthal angle [0, 2π]
    cos_theta = 2 * np.random.rand() - 1  # uniform in [-1, 1]
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand()
    
    # calculate displacement components
    dx = step_distance * sin_theta * np.cos(phi)
    dy = step_distance * sin_theta * np.sin(phi)
    dz = step_distance * cos_theta
    
    # updating the position
    x_new = x_curr + dx
    y_new = y_curr + dy
    z_new = z_curr + dz
    
    solar_x_positions.append(x_new)
    solar_y_positions.append(y_new)
    solar_z_positions.append(z_new)
    
    # update total distance and time
    solar_total_dist += step_distance
    solar_time = (solar_total_dist * u.m) / const.c
    
    # new radius
    r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)
    
    # check if escaped (at 0.9 R_sun, density drops and scattering stops)
    if r_new >= 0.9 * R_sun_m:
        solar_sim_exit = True
        return r_new, True
    
    return r_new, False

def solar_init():
    """Initialize 3D plot elements for animation."""
    solar_line.set_data(np.array([]), np.array([]))
    solar_line.set_3d_properties(np.array([]))
    solar_scat_point.set_data(np.array([]), np.array([]))
    solar_scat_point.set_3d_properties(np.array([]))
    solar_current_point.set_data(np.array([]), np.array([]))
    solar_current_point.set_3d_properties(np.array([]))
    return solar_line, solar_scat_point, solar_current_point, stats_text

def update_solar_animation(frame):
    """
    Animation update function - called for each frame.
    Performs one scatter and updates all plot elements.
    """
    global solar_x_positions, solar_y_positions, solar_z_positions
    global solar_total_dist, solar_sim_exit, solar_time
    
    if solar_sim_exit:
        return solar_line, solar_scat_point, solar_current_point, stats_text
    
    # perform scatter
    radius, escaped = scatter_photon_3d()
    
    # update photon path line
    solar_line.set_data(np.array(solar_x_positions), np.array(solar_y_positions))
    solar_line.set_3d_properties(np.array(solar_z_positions))
    
    # update scatter points (all except current)
    if len(solar_x_positions) > 1:
        solar_scat_point.set_data(
            np.array(solar_x_positions[:-1]), 
            np.array(solar_y_positions[:-1])
        )
        solar_scat_point.set_3d_properties(np.array(solar_z_positions[:-1]))
    
    # update current position marker
    curr_x = solar_x_positions[-1]
    curr_y = solar_y_positions[-1]
    curr_z = solar_z_positions[-1]
    solar_current_point.set_data([curr_x], [curr_y])
    solar_current_point.set_3d_properties([curr_z])
    
    # update title
    mfp_display = compute_mfp_at_radius(radius)
    ax.set_title(
        f'Photon Scattering Inside Sun | Scatters: {len(solar_x_positions) - 1} | MFP = {mfp_display:.2e} m',
        fontsize=14
    )
    
    # dynamic camera following photon
    if not solar_sim_exit:
        zoom_factor = 0.35
        x_center, y_center, z_center = curr_x, curr_y, curr_z
        window_size = R_sun_m * zoom_factor
        
        ax.set_xlim([x_center - window_size, x_center + window_size])
        ax.set_ylim([y_center - window_size, y_center + window_size])
        ax.set_zlim([z_center - window_size, z_center + window_size])
        
        azim = 30 + frame * 3
        elev = 25 + 10 * np.sin(frame * 0.1)
        ax.view_init(elev=elev, azim=azim)
    
    # update statistics text
    r_in_rsun = (radius * u.m).to(u.R_sun).value
    time_us = solar_time.to(u.microsecond).value
    stats_text.set_text(
        f'Distance: {solar_total_dist:.2e} m\n'
        f'Radius: {r_in_rsun:.3f} R☉\n'
        f'Time: {time_us:.2e} μs'
    )
    
    return solar_line, solar_scat_point, solar_current_point, stats_text

# create 3D figure
solar_fig = plt.figure(figsize=(12,12))
ax = solar_fig.add_subplot(111, projection='3d')

# create Sun sphere surface
u_sphere = np.linspace(0, 2 * np.pi, 80)
v_sphere = np.linspace(0, np.pi, 80)
x_sphere = R_sun_m * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_sphere = R_sun_m * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_sphere = R_sun_m * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))

# plot Sun
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.5, edgecolor='none')
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gold', alpha=0.15, linewidth=0.3)

# create plot elements for photon
solar_line, = ax.plot3D([], [], [], 'k-', linewidth=1.3, label='Photon Path')
solar_scat_point, = ax.plot3D([], [], [], 'o', color='crimson', markersize=5, label='Scatter Events')
solar_current_point, = ax.plot3D([], [], [], 'o', color='lime', markersize=9, label='Current Position')

# stats text box
stats_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

# configure axes
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('Radiative Transfer in the Sun', fontsize=15, fontweight='bold')
ax.set_box_aspect([1, 1, 1])
ax.legend(loc='upper right', fontsize=10)

# set axis limits
limit = R_sun_m * 1.15
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# create animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

solar_anim = FuncAnimation(
    solar_fig,
    update_solar_animation,
    frames=range(solar_max_scatters),
    init_func=solar_init,
    blit=False,
    interval=solar_time_delay,
    repeat=False
)
plt.show()
# debugged with Claude.ai
