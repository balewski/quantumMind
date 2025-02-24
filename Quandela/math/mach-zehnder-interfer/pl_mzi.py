import matplotlib.pyplot as plt

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Draw light source to Beam Splitter 1 (BS1)
ax.plot([0, 1], [2, 2], 'k-', label='Photon Path')  # Horizontal line from Source to BS1
ax.text(0, 2.1, 'Source', fontsize=12)

# Draw Beam Splitter 1
ax.plot([1, 1.5], [2, 2.5], 'k--')  # Path A (up)
ax.plot([1, 1.5], [2, 1.5], 'k--')  # Path B (down)
ax.plot(1, 2, 'ro', label='Beam Splitter 1')
ax.text(1, 2.1, 'BS1', fontsize=12)

# Draw Paths A and B to mirrors
ax.plot([1.5, 3], [2.5, 2.5], 'k-', label='Path A')  # Path A (upper)
ax.plot([1.5, 3], [1.5, 1.5], 'k-', label='Path B')  # Path B (lower)

# Draw Mirrors
ax.plot(3, 2.5, 'ro', label='Mirror A')
ax.plot(3, 1.5, 'ro', label='Mirror B')
ax.text(3, 2.6, 'Mirror A', fontsize=12)
ax.text(3, 1.4, 'Mirror B', fontsize=12)

# Reflect Paths towards Beam Splitter 2 (BS2)
ax.plot([3, 4.5], [2.5, 2], 'k--')  # Path A reflected
ax.plot([3, 4.5], [1.5, 2], 'k--')  # Path B reflected

# Draw Beam Splitter 2 (BS2)
ax.plot(4.5, 2, 'ro', label='Beam Splitter 2')
ax.text(4.5, 2.1, 'BS2', fontsize=12)

# Draw paths to Detectors C and D
ax.plot([4.5, 5.5], [2, 2.5], 'k-', label='To Detector C')  # To Detector C
ax.plot([4.5, 5.5], [2, 1.5], 'k-', label='To Detector D')  # To Detector D
ax.text(5.6, 2.5, 'Detector C', fontsize=12)
ax.text(5.6, 1.5, 'Detector D', fontsize=12)

# Labels for quantum states on paths
ax.text(2.25, 2.6, '| A ⟩', fontsize=14, color='blue')
ax.text(2.25, 1.4, '| B ⟩', fontsize=14, color='green')

# Draw quantum states as vertical dashed lines
ax.axvline(x=0, color='gray', linestyle='--')
ax.text(0, 1.2, r'$|\psi_0\rangle$', fontsize=14, color='purple')

ax.axvline(x=1, color='gray', linestyle='--')
ax.text(1, 1.2, r'$|\psi_1\rangle$', fontsize=14, color='purple')

ax.axvline(x=3, color='gray', linestyle='--')
ax.text(3, 1.2, r'$|\psi_2\rangle$', fontsize=14, color='purple')

ax.axvline(x=4.5, color='gray', linestyle='--')
ax.text(4.5, 1.2, r'$|\psi_3\rangle$', fontsize=14, color='purple')

# General plot settings
ax.set_xlim(-0.5, 6)
ax.set_ylim(1, 3)
ax.axis('off')
ax.set_title('Mach-Zehnder Interferometer Diagram with Quantum States')

# Show the diagram
plt.show()
