# Molecule-Pathtracer-Master-Thesis-PUBLIC
## Objective: Render a large number (50k) of primitives using progressive path tracing.
1. I have explored various acceleration structures and their variants.
2. I worked on optimizing the layout for BVH nodes in the GPU buffer to achieve more efficient cache hits.
3. Maintaining Real-time Performance: 1 shader = 1 ray bounce: The shader doesn't compute the entire path but only calculates one bounce at a time. Each subsequent bounce is calculated by another instance of the shader program. The necessary information is stored in textures. If a shader instance were to compute the entire path (e.g., 10 bounces), the user would have to wait if they wanted to move the camera

## TODO:

- solve fireflies issue
- resolve memory leaks.
- create macros for file paths.
- UI: selection of protein, colors, skybox.

High Roughness: https://youtu.be/Qd9NzgdKY5w
Zero Roughness: https://youtu.be/kDUHRfjC6WU