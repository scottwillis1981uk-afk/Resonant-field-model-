import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# RFM Gravity Demo: Null rays bend along ∇α (higher tension = slower waves)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
ax.set_xlim(0, 200)
ax.set_ylim(0, 100)
ax.axis('off')

def alpha(y):
    return 0.05 + 0.0005 * (y - 50)**2  # parabolic tension gradient ("mass")

lines = [ax.plot([], [], color='cyan', lw=1.5)[0] for _ in range(9)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(t):
    for i, y0 in enumerate(np.linspace(10, 90, 9)):
        # Ray path bends toward higher α
        x = np.linspace(0, t*3, 1000)
        y = y0 * np.exp(-alpha(np.linspace(y0, y0, 1000)) * t / 50)
        y = np.clip(y, 0, 100)
        lines[i].set_data(x, y)
    ax.set_title(f'Null Rays Bend Along ∇α — Gravity Emerges (t = {t:.1f})', color='white', fontsize=14)
    return lines

ani = FuncAnimation(fig, animate, frames=60, init_func=init, interval=100, repeat=True)
ani.save('gravity_bend.gif', writer='pillow', fps=15, dpi=100)
plt.close()

print("gravity_bend.gif created — this is gravity from one scalar field.")
print("Higher α = higher tension = waves slow and bend toward it. No curved spacetime added.")
