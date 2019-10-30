from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from math import cos, sin
from simple_camera import mesh_core_cython


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3].
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype=np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


## -------------- Camera. from world space to camera space
# Ref: https://cs184.eecs.berkeley.edu/lecture/transforms-2
def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    norm = np.maximum(norm, epsilon)
    return x / norm


def lookat_camera(vertices, eye, at=None, up=None):
    """ 'look at' transformation: from world space to camera space
    standard camera space:
        camera located at the origin.
        looking down negative z-axis.
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    Args:
      vertices: [nver, 3]
      eye: [3,] the XYZ world space position of the camera.
      at: [3,] a position along the center of the camera's gaze.
      up: [3,] up direction
    Returns:
      transformed_vertices: [nver, 3]
    """
    if at is None:
        at = np.array([0, 0, 0], np.float32)
    if up is None:
        up = np.array([0, 1, 0], np.float32)

    eye = np.array(eye).astype(np.float32)
    at = np.array(at).astype(np.float32)
    z_aixs = -normalize(at - eye)  # look forward
    x_aixs = normalize(np.cross(up, z_aixs))  # look right
    y_axis = np.cross(z_aixs, x_aixs)  # look up

    R = np.stack((x_aixs, y_axis, z_aixs))  # , axis = 0) # 3 x 3
    transformed_vertices = vertices - eye  # translation
    transformed_vertices = transformed_vertices.dot(R.T)  # rotation
    return transformed_vertices


def perspective_project(vertices, fovy, aspect_ratio=1., near=0.1, far=1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3]
    '''
    fovy = np.deg2rad(fovy)
    top = near * np.tan(fovy)
    bottom = -top
    right = top * aspect_ratio
    left = -right

    # -- homo
    P = np.array([[near / right, 0, 0, 0],
                  [0, near / top, 0, 0],
                  [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                  [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # [nver, 4]
    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices / projected_vertices[:, 3:]
    projected_vertices = projected_vertices[:, :3]
    projected_vertices[:, 2] = -projected_vertices[:, 2]

    # -- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices


def to_image(vertices, h, w, is_perspective=False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:, 0] = image_vertices[:, 0] * w / 2
        image_vertices[:, 1] = image_vertices[:, 1] * h / 2
    # move to center of image
    image_vertices[:, 0] = image_vertices[:, 0] + w / 2
    image_vertices[:, 1] = image_vertices[:, 1] + h / 2
    # flip vertices along y-axis.
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1
    return image_vertices


def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :]  # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :]  # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :]  # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2)  # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices, dtype=np.float32).copy()  # [nver, 3]
    # for i in range(triangles.shape[0]):
    #     normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
    #     normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
    #     normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
    mesh_core_cython.get_normal_core(normal, tri_normal.astype(np.float32).copy(), triangles.copy(), triangles.shape[0])

    # normalize to unit length
    mag = np.sum(normal ** 2, 1)  # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal / np.sqrt(mag[:, np.newaxis])

    return normal


def add_light(vertices, triangles, colors, light_positions=0, light_intensities=0):
    ''' Gouraud shading. add point lights.
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    3. No specular (unless skin is oil, 23333)
    Ref: https://cs184.eecs.berkeley.edu/lecture/pipeline
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        light_positions: [nlight, 3]
        light_intensities: [nlight, 3]
    Returns:
        lit_colors: [nver, 3]
    '''
    nver = vertices.shape[0]
    normals = get_normal(vertices, triangles)  # [nver, 3]

    # ambient
    # La = ka*Ia

    # diffuse
    # Ld = kd*(I/r^2)max(0, nxl)
    direction_to_lights = vertices[np.newaxis, :, :] - light_positions[:, np.newaxis, :]  # [nlight, nver, 3]
    direction_to_lights_n = np.sqrt(np.sum(direction_to_lights ** 2, axis=2))  # [nlight, nver]
    direction_to_lights = direction_to_lights / direction_to_lights_n[:, :, np.newaxis]
    normals_dot_lights = normals[np.newaxis, :, :] * direction_to_lights  # [nlight, nver, 3]
    normals_dot_lights = np.sum(normals_dot_lights, axis=2)  # [nlight, nver]
    diffuse_output = colors[np.newaxis, :, :] * normals_dot_lights[:, :, np.newaxis] * light_intensities[:, np.newaxis,
                                                                                       :]
    diffuse_output = np.sum(diffuse_output, axis=0)  # [nver, 3]

    # specular
    # h = (v + l)/(|v + l|) bisector
    # Ls = ks*(I/r^2)max(0, nxh)^p
    # increasing p narrows the reflectionlob

    lit_colors = diffuse_output  # only diffuse part here.
    lit_colors = np.minimum(np.maximum(lit_colors, 0), 1)
    return lit_colors


def render_colors_fast(vertices, triangles, colors, h, w, c = 3, BG = None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    '''

    # initial
    if BG is None:
        image = np.zeros((h, w, c), dtype = np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    ###
    mesh_core_cython.render_colors_core(
                image, vertices, triangles,
                colors,
                depth_buffer,
                vertices.shape[0], triangles.shape[0],
                h, w, c)
    return image