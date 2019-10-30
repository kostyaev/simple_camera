from simple_camera.ops import *


def render_perspective_camera(vertices, faces, w=256, h=256,
                              light_positions=np.array([[0, 0, -100]]),
                              light_intensities=np.array([[1.2, 1.2, 1.2]]),
                              camera_pos=(0, 0, 250)
                              ):
    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    s = 180 / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    R = angle2matrix([0, 0, 0])
    t = [0, 0, 0]
    vertices = similarity_transform(vertices, s, R, t)  # transformed vertices
    camera_vertices = lookat_camera(vertices, eye=camera_pos, at=[0, 0, 0], up=[0, 1, 0])
    projected_vertices = perspective_project(camera_vertices, 15, near=1000, far=-100)
    image_vertices = to_image(projected_vertices, h, w, is_perspective=True)

    colors = np.ones_like(image_vertices) * 0.5

    light_colors = add_light(vertices, faces, colors,
                             light_positions=light_positions,
                             light_intensities=light_intensities)

    rendering = render_colors_fast(image_vertices, faces, light_colors, h, w)
    rendering = (rendering.clip(0, 1) * 255).astype(np.uint8)

    return rendering