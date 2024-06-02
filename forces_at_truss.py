import numpy as np
import matplotlib.pyplot as plt

# Geometry of the truss
points = {
    'A': np.array([0, 0]),
    'H': np.array([5, 0]),
    'B': np.array([5, 4]),
    'G': np.array([10, 0]),
    'C': np.array([10, 4]),
    'F': np.array([15, 0]),
    'D': np.array([15, 4]),
    'E': np.array([20, 0])
}

# External loads (in kN)
loads = {
    'A': np.array([0, 0]),
    'H': np.array([0, -30]),
    'B': np.array([0, 0]),
    'G': np.array([0, -60]),
    'C': np.array([0, 0]),
    'F': np.array([0, -30]),
    'D': np.array([0, 0]),
    'E': np.array([0, 0])
}

# Member connectivity
members = {
    'AB': ('A', 'B'),
    'AH': ('A', 'H'),
    'BH': ('B', 'H'),
    'BC': ('B', 'C'),
    'CD': ('C', 'D'),
    'DE': ('D', 'E'),
    'DF': ('D', 'F'),
    'CH': ('C', 'H'),
    'CG': ('C', 'G'),
    'HF': ('H', 'F'),
    'GF': ('G', 'F'),
    'EF': ('E', 'F'),
    'CF': ('C', 'F'),
    'GH': ('G', 'H')
}

# Create the equilibrium equations
# 16 equations for 16 unknowns (13 member forces + 3 reactions: Ax, Ay, Ey)
num_joints = len(points)
num_members = len(members)
A = np.zeros((2 * num_joints, num_members + 3))  # Extra columns for Ax, Ay, Ey
b = np.zeros(2 * num_joints)

# Angles of members
angles = {}
for member, (start, end) in members.items():
    dx = points[end][0] - points[start][0]
    dy = points[end][1] - points[start][1]
    length = np.sqrt(dx**2 + dy**2)
    angles[member] = (dx / length, dy / length)

# Joint equilibrium equations
member_list = list(members.keys())
for i, joint in enumerate(points.keys()):
    row_x = 2 * i
    row_y = 2 * i + 1
    for j, member in enumerate(member_list):
        start, end = members[member]
        if joint == start:
            A[row_x, j] = angles[member][0]
            A[row_y, j] = angles[member][1]
        elif joint == end:
            A[row_x, j] = -angles[member][0]
            A[row_y, j] = -angles[member][1]
    if joint == 'A':
        A[row_x, num_members] = 1  # Ax
        A[row_y, num_members + 1] = 1  # Ay
    if joint == 'E':
        A[row_y, num_members + 2] = 1  # Ey

# External loads
for i, joint in enumerate(loads.keys()):
    b[2 * i:2 * i + 2] = loads[joint]

# Solve the system of equations
forces_reactions = np.linalg.lstsq(A, b, rcond=None)[0]

# Extract forces in members and reactions
forces = forces_reactions[:num_members]
reactions = forces_reactions[num_members:]

# Force magnitudes for color coding
forces_magnitude = {member: abs(forces[i]) for i, member in enumerate(member_list)}

# Plot the truss
fig, ax = plt.subplots()
for member, (start, end) in members.items():
    x_values = [points[start][0], points[end][0]]
    y_values = [points[start][1], points[end][1]]
    force_magnitude = forces_magnitude[member]
    color = plt.cm.viridis(force_magnitude / max(forces_magnitude.values()))
    ax.plot(x_values, y_values, color=color, label=f'{member} ({force_magnitude:.2f} kN)')

# Plot joints
for point, coord in points.items():
    ax.plot(coord[0], coord[1], 'ko')
    ax.text(coord[0], coord[1], f' {point}', verticalalignment='bottom')

# Plot the external loads
for joint, load in loads.items():
    if np.any(load != 0):
        ax.arrow(points[joint][0], points[joint][1], load[0] / 10, load[1] / 10,
                 head_width=0.3, head_length=0.5, fc='r', ec='r')

# Annotations and labels
ax.legend()
ax.set_title('Truss Analysis')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
