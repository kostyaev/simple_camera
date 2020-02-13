from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from simple_camera.ops import *


def render_perspective_camera(vertices, faces, width=256, height=256,
                              angles=(0, 0, 0), translation=(0, 0, 0), scale=1.0,
                              light_positions=np.array([[0, 0, -100]]),
                              light_intensities=np.array([[1.2, 1.2, 1.2]])):
    """
    Renders an image using perspective camera.
    Args:
        vertices: numpy array of shape [nb_vertices, 3].
        faces: numpy integer array of shape [nb_faces, 3].
        width: width of the rendered image.
        height: height of the rendered image.
        angles: [pitch, yaw, roll] degrees of rotations.
        scale: A float value, scale multiplier applied to vertices coordinates.

    Returns:
        A Numpy array representing an uint8 rgb image with shape [height,width,3]

    """
    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    s = 180 * scale
    R = angle2matrix(angles)
    t = translation
    vertices = similarity_transform(vertices, s, R, t)  # transformed vertices
    camera_vertices = lookat_camera(vertices, eye=(0, 0, 250), at=[0, 0, 0], up=[0, 1, 0])
    projected_vertices = perspective_project(camera_vertices, fovy=15, near=1000, far=-100)
    image_vertices = to_image(projected_vertices, height, width, is_perspective=True)

    colors = np.ones_like(image_vertices) * 0.5
    light_colors = add_light(vertices, faces, colors,
                             light_positions=light_positions,
                             light_intensities=light_intensities)

    rendering = render_colors_fast(image_vertices, faces, light_colors, height, width)
    rendering = (rendering.clip(0, 1) * 255).astype(np.uint8)

    return rendering


def texture_render_perspective_camera(vertices, faces, texture, tex_coords, tex_triangles,
                                      width=256, height=256, angles=(0, 0, 0),
                                      translation=(0, 0, 0), scale=1.0):
    """
    Renders an image using perspective camera.
    Args:
        vertices: numpy array of shape [nb_vertices, 3].
        faces: numpy integer array of shape [nb_faces, 3].
        width: width of the rendered image.
        height: height of the rendered image.
        angles: [pitch, yaw, roll] degrees of rotations.
        scale: A float value, scale multiplier applied to vertices coordinates.

    Returns:
        A Numpy array representing an uint8 rgb image with shape [height,width,3]

    """
    vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
    s = 180 * scale / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    R = angle2matrix(angles)
    t = translation
    vertices = similarity_transform(vertices, s, R, t)  # transformed vertices
    camera_vertices = lookat_camera(vertices, eye=(0, 0, 250), at=[0, 0, 0], up=[0, 1, 0])
    projected_vertices = perspective_project(camera_vertices, fovy=15, near=1000, far=-100)
    image_vertices = to_image(projected_vertices, height, width, is_perspective=True)

    rendering = render_texture_fast(image_vertices, faces, texture, tex_coords, tex_triangles, h=height, w=width)
    rendering = (rendering * 255).clip(0, 255).astype(np.uint8)

    return rendering
