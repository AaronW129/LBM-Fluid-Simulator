import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.widgets import Slider


def main():
    # --- 0. INITIALIZATIONS ---
    tick = 10
    grid_x, grid_y = 300, 100 # size of lattice
    tau = 0.8 # collision time
    end_tick = 100000 # how many ticks before simulation ends     
    weight_count = 9 
    txt_naca = {"value":"2412"} # explicitely make the NACA code mutable

    # -- WEIGHTS
    # Direction of weight per cell
    e = np.array([
        [-1,  1], [0,  1], [1,  1],
        [-1,  0], [0,  0], [1,  0],
        [-1, -1], [0, -1], [1, -1]
    ], dtype=np.int32)

    # Weight per direction 
    w = np.array([
        1/36, 1/9,  1/36,
        1/9,  4/9,  1/9,
        1/36, 1/9,  1/36
    ], dtype=np.float64)

    # Corresponds to the OPPOSITE vector (ex. 1 opp. to 7)
    opp = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])

    # -- LATTICE
    # Lattice with weight values per cell (properly initizalied in collisions section)
    fluid_grid = np.zeros((grid_y, grid_x, weight_count), dtype=np.float64)

    # -- FLOW
    # Initial flow (polar coordinates)
    r_in = 0.12
    theta_in = 0

    # Flow vector to be mapped on each of lattice's cells
    u0_mag = r_in
    u0_dir = np.array([np.cos(theta_in), np.sin(theta_in)], dtype=np.float64)
    if np.linalg.norm(u0_dir) == 0:
        u0_dir = np.array([1.0, 0.0], dtype=np.float64)

    # System-wide attributes to be mapped onto each cell
    ux0 = np.full((grid_y, grid_x), u0_mag * u0_dir[0], dtype=np.float64)
    uy0 = np.full((grid_y, grid_x), u0_mag * u0_dir[1], dtype=np.float64)
    rho0 = np.ones((grid_y, grid_x), dtype=np.float64)

    # --- COLLISIONS ---
    # Returns cell's densities in thermo-equilibrium
    def equilibrium(rho: float, ux: float, uy: float) -> float:
        u_sq = ux**2 + uy**2
        feq = np.zeros((grid_y, grid_x, 9), dtype=np.float64)
        for i in range(9):
            eu = e[i, 0] * ux + e[i, 1] * uy
            feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u_sq)
        return feq
   
    # Returns the cell's macroscopic properties, derived from microscopic density
    def macroscopic(fluid_grid) -> tuple[float, float, float]:
        rho = np.sum(fluid_grid, axis=2)
        ux = np.sum(fluid_grid * e[:, 0], axis=2) / rho
        uy = np.sum(fluid_grid * e[:, 1], axis=2) / rho
        return rho, ux, uy

    # Initialize lattice with random noise in fluid
    fluid_grid[:, :, :] = equilibrium(rho0, ux0, uy0) + 0.01 * np.random.randn(grid_y, grid_x, weight_count)

    # --- OBSTACLES & ROTATIONS ---
    # Obstacle coordinates (DEFAULT)
    X, Y = np.meshgrid(np.arange(grid_x), np.arange(grid_y))
    obstacle = np.zeros((grid_y, grid_x), dtype=bool) # if 0, it is a fluid. if 1, solid
    obstacle_type = {"name": "circle", "center": [grid_x // 4, grid_y // 2], "angle": 0.0, "dragging": False}

    def rotate_mask(shape: np.ndarray, center: tuple[float, float], angle: float):
        if angle == 0.0:
            return shape
        
        cx, cy = center
        ys, xs = np.where(shape) # current shape coordinates

        # Shift shape to origin
        x = xs - cx
        y = ys - cy

        # Apply rotation via rotation matrix [cos, sin ; -sin, cos]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        xr = cos_a * x - sin_a * y
        yr = sin_a * x + cos_a * y

        # Bring rotated shape back to grid
        xs_new = np.round(xr + cx).astype(int)
        ys_new = np.round(yr + cy).astype(int)

        rotated = np.zeros_like(shape)
        valid = (
            (xs_new >= 0) & (xs_new < grid_x) &
            (ys_new >= 0) & (ys_new < grid_y)
        )
        rotated[ys_new[valid], xs_new[valid]] = True
        return rotated


    # -- SHAPE GENERATORS
    # (x-x0)^2 + (y-y0)^2 <= r^2
    def make_circle(center: tuple) -> np.ndarray:
        x0, y0 = center
        r = grid_y / 4
        return (X - x0)**2 + (Y - y0)**2 < r**2

    # |x-x0| + |y-y0| <= half-side-length
    def make_square(center: tuple) -> np.ndarray:
        x0, y0 = center
        half = grid_y // 4 
        return (np.abs(X - x0) <= half) & (np.abs(Y - y0) <= half)
    
    # vertical wall at center x with a small vertical hole
    def make_wall_with_gap(center: tuple) -> np.ndarray:
        x0, y0 = center
        half_thick = 3
        gap = grid_y / 4
        band = (X >= x0 - half_thick) & (X <= x0 + half_thick)
        slit = (np.abs(Y - y0) <= gap / 2)
        return band & (~slit)

    # Creates the naca airfoil from its NACA ID
    def make_naca_airfoil(center: tuple) -> np.ndarray:
        code = txt_naca["value"]
        cx, cy = center
        # Parse code
        m = int(code[0]) / 100      # max camber
        p = int(code[1]) / 10       # max camber location
        t = int(code[2:]) / 100     # thickness

        # Chord length relative to grid
        chord = int(grid_y * 0.8)

        # Normalized x coordinate along chord
        x0 = cx - chord // 2
        xi = (X - x0) / chord
        inside = (xi >= 0.0) & (xi <= 1.0)
        xic = np.clip(xi, 0.0, 1.0)

        # Thickness distribution
        yt = 5.0 * t * (
            0.2969 * np.sqrt(xic) - 0.1260 * xic
            - 0.3516 * xic**2 + 0.2843 * xic**3 - 0.1015 * xic**4
        )

        # Camber line and slope
        if p == 0 or m == 0:
            yc = np.zeros_like(xi)
            dyc_dx = np.zeros_like(xi)
        else:
            yc = np.where(
                xi < p,
                m / (p**2) * (2 * p * xi - xi**2),
                m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xi - xi**2),
            )
            dyc_dx = np.where(
                xi < p,
                2 * m / (p**2) * (p - xi),
                2 * m / ((1 - p) ** 2) * (p - xi),
            )

        theta = np.arctan(dyc_dx)

        # yical bounds of airfoil section
        y_upper = (yc + yt * np.cos(theta)) * chord
        y_lower = (yc - yt * np.cos(theta)) * chord

        # Map onto lattice
        y_local = cy - Y
        foil = inside & (y_local <= y_upper) & (y_local >= y_lower)
        return foil

    # generates the corresponding obstacle with appropriate rotation
    def generate_obstacle() -> np.ndarray:
        name = obstacle_type["name"]
        center = obstacle_type["center"]
        angle = obstacle_type["angle"]

        obstacle[0, :] = True
        obstacle[-1, :] = True

        if name == "circle":
            base = make_circle(center)
        elif name == "square":
            base = make_square(center)
        elif name == "hydro":
            base = make_naca_airfoil(center)
        elif name == "wallgap":
            base = make_wall_with_gap(center)
        else:
            base = np.zeros_like(obstacle, dtype=bool)

        return rotate_mask(base, center, angle) 

    # Create obstacle and walls
    obstacle[:] = generate_obstacle()

    # --- GRAPH PARAMETERS ---
    # Figure and UI
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.4,top=0.8)  # space for two rows of buttons
    im = ax.imshow(np.zeros((grid_y, grid_x)), cmap="viridis", origin="upper", interpolation="nearest", vmin = 0.0, vmax = 0.40)
    plt.colorbar(im, pad=0.04, label="|u| (fluid speed)")
    title = ax.set_title("LBM wind-tunnel, tick=0")

    # --- BUTTONS ---
    # Padding
    x_size = 0.2
    y_size = 0.1
    y_spacing = 0.05
    x_spacing = 0.23
    
    # Placement of action buttons (bottom-row)
    ax_reset = plt.axes([0.05, y_spacing, x_size, y_size])
    ax_play = plt.axes([0.05 + x_spacing, y_spacing, x_size, y_size])
    ax_pause = plt.axes([0.05 + (x_spacing * 2), y_spacing, x_size, y_size])
    ax_naca = plt.axes([0.05 + (x_spacing * 3 + x_size / 2), y_spacing, x_size / 2, y_size])

    # Placement of shape buttons (top-row)
    ax_circle = plt.axes([0.05, y_spacing * 4, x_size, y_size])
    ax_square = plt.axes([0.05 + x_spacing, y_spacing * 4, x_size, y_size])
    ax_hydro = plt.axes([0.05 + (x_spacing * 2), y_spacing * 4, x_size, y_size])
    ax_wallgap = plt.axes([0.05 + (x_spacing * 3), y_spacing * 4, x_size, y_size])

    # Placement of angle slider (above plot)
    ax_angle = plt.axes([0.05 + x_spacing, y_spacing * 18, 0.3, 0.04])
    slider_angle = Slider(
        ax=ax_angle,
        label="Angle of attack (deg)",
        valmin=-180,
        valmax=180,
        valinit=0,
        valstep=1
    )

    # Style of buttons
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.85")
    btn_play = Button(ax_play, "Play", hovercolor="0.85")
    btn_pause = Button(ax_pause, "Pause", hovercolor="0.85")
    box_naca = TextBox(ax_naca, "NACA: ", initial="2412", textalignment="center",)

    # Shape buttons labels
    btn_circle = Button(ax_circle, "Circle", hovercolor="0.85")
    btn_square = Button(ax_square, "Square", hovercolor="0.85")
    btn_hydro = Button(ax_hydro, "Airfoil", hovercolor="0.85")
    btn_wallgap = Button(ax_wallgap, "Wall + Gap", hovercolor="0.85")


    # --- SIMULATION TIME LOOP ---
    # State and timer
    state = {"t": 0, "running": False}
    timer = fig.canvas.new_timer(interval=10)

    def update_visual():
        _, ux, uy = macroscopic(fluid_grid)
        speed = np.sqrt(ux**2 + uy**2)
        speed = np.ma.array(speed, mask=obstacle)
        im.set_data(speed)
        title.set_text(f"LBM wind-tunnel, tick={state['t']}")
        fig.canvas.draw_idle()

    def step():
        substeps = 2
        nonlocal fluid_grid
        for _ in range(substeps):
            if state["t"] >= end_tick:
                timer.stop()
                state["running"] = False
                return

            # Macroscopic states
            rho, ux, uy = macroscopic(fluid_grid)

            # Collision
            feq = equilibrium(rho, ux, uy)
            fluid_grid[:] = fluid_grid - (1 / tau) * (fluid_grid - feq)

            # Streaming (periodic)
            for i in range(9):
                fluid_grid[:, :, i] = np.roll(np.roll(fluid_grid[:, :, i], e[i, 0], axis=1), e[i, 1], axis=0)

            # Zou-He boundaries on left and right walls
            fluid_grid[:, -1, :] = fluid_grid[:, -2, :]
            fluid_grid[:, 0, :] = fluid_grid[:, 1, :]
            
            # Bounce-back (ceiling and floor)
            fluid_grid[0, :, [2, 5, 8]] = fluid_grid[0, :, [6, 3, 0]]
            fluid_grid[-1, :, [6, 3, 0]] = fluid_grid[-1, :, [2, 5, 8]]

            # Bounce-back on obstacle
            f_post = fluid_grid.copy()
            for i in range(9):
                fluid_grid[obstacle, i] = f_post[obstacle, opp[i]]

            # Move forward in time
            state["t"] += 1

        # Update visual if the time is 
        if state["t"] % tick == 0:
            update_visual()

    timer.add_callback(step)

    # --- BUTTONS ACTIONS ---
    # Function to change shape
    def set_shape(name):
        obstacle_type["name"] = name
        obstacle_type["center"] = [grid_x // 4, grid_y // 2]
        obstacle[:] = generate_obstacle()
        update_visual()

    # Button actions
    def on_play(event):
        if not state["running"]:
            timer.start()
            state["running"] = True

    def on_pause(event):
        if state["running"]:
            timer.stop()
            state["running"] = False

    def on_reset(event):
        # Stop and reset time
        timer.stop()
        state["running"] = False
        state["t"] = 0
        # Reinitialize fields
        ux0[:] = u0_mag * u0_dir[0]
        uy0[:] = u0_mag * u0_dir[1]
        rho0[:] = 1.0
        fluid_grid[:, :, :] = equilibrium(rho0, ux0, uy0) + 0.01 * np.random.randn(grid_y, grid_x, weight_count)
        # Reset obstacle center and mask
        obstacle_type["center"] = [grid_x // 4, grid_y // 2]
        obstacle[:] = generate_obstacle()
        update_visual()

    def on_naca_submit(text):
        text = text.strip()

        # Basic validation: NACA 4-digit
        if not text.isdigit() or len(text) != 4:
            print("Invalid NACA code (must be 4 digits)")
            return

        txt_naca["value"] = text

        # Only regenerate if airfoil is selected
        if obstacle_type["name"] == "hydro":
            obstacle[:] = generate_obstacle()
            update_visual()

    def on_angle_change(val):
        obstacle_type["angle"] = np.deg2rad(val)
        obstacle[:] = generate_obstacle()
        update_visual()

    # When shape button pressed, create shape
    def on_circle(event): set_shape("circle")
    def on_square(event): set_shape("square")
    def on_hydro(event): set_shape("hydro")
    def on_wallgap(event): set_shape("wallgap")
    

    # When the BUTTON is clicked, do the following options.
    btn_reset.on_clicked(on_reset)
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    btn_circle.on_clicked(on_circle)
    btn_square.on_clicked(on_square)
    btn_hydro.on_clicked(on_hydro)
    btn_wallgap.on_clicked(on_wallgap)

    # When the TEXT is submitted, do the following operations:
    box_naca.on_submit(on_naca_submit)

    # When the SLIDER is edited, do the following operations:
    slider_angle.on_changed(on_angle_change)

    # --- CLICK AND DRAG EVENTS ---
    # Mouse interaction: click-and-drag obstacle
    def on_press(event):
        # if the click is not in the graph or there is no object here, don't do anything
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        # extract the x, y coordinates of the mouse
        x = int(np.clip(event.xdata, 0, grid_x - 1))
        y = int(np.clip(event.ydata, 0, grid_y - 1))
        if obstacle[y, x]:
            obstacle_type["dragging"] = True

    def on_release(event):
        obstacle_type["dragging"] = False

    def on_motion(event):
        if not obstacle_type["dragging"]:
            return
        # Have the object's center be at the mouse's cursor
        x = int(np.clip(event.xdata, 0, grid_x))
        y = int(np.clip(event.ydata, 0, grid_y))
        obstacle_type["center"] = [x, y]
        obstacle[:] = generate_obstacle()
        update_visual()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # Start paused with initial visual
    update_visual()
    plt.show()

if __name__ == "__main__":
    main()
