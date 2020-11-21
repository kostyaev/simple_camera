from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from simple_camera.ops import *


def render_perspective_camera(vertices, faces, width=256, height=256,
                              angles=(0, 0, 0), translation=(0, 0, 0), scale=1.0,
                              light_positions=np.array([[0, 0, -100]]),
                              light_intensities=np.array([[1.2, 1.2, 1.2]]),
                              bg_img=None):
    """
    Renders an image using perspective camera.
    Args:
        vertices: numpy array of shape [nb_vertices, 3].
        faces: numpy integer array of shape [nb_faces, 3].
        width: width of the rendered image.
        height: height of the rendered image.
        angles: [pitch, yaw, roll] degrees of rotations.
        translation: 3D vector of translation,
        light_positions: nx3 array, each row is a 3D vector describing light position
        light_intensities: nx3 array, each row is a 3D vector describing light intensity
        scale: A float value, scale multiplier applied to vertices coordinates.
        bg_img: numpy array representing background image of size [height,width,3] with 0..1 float values
    Returns:
        A Numpy array representing an uint8 rgb image with shape [height,width,3]

    """
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    assert vertices.ndim == 2 and faces.ndim == 2

    light_positions = np.array(light_positions, dtype=np.float32)
    light_intensities = np.array(light_intensities, dtype=np.float32)
    assert light_positions.ndim == 2 and light_intensities.ndim == 2

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

    rendering = render_colors_fast(image_vertices, faces, light_colors, height, width, BG=bg_img)
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
