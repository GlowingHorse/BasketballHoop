import numpy as np
import math
import matplotlib.pyplot as plt
import random
from stl import mesh
import pickle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

G = 9.8  # m/s^2，不知道单位对不对


def compute_angle_to_hit_hoop(bbpos_inner, hpos, iv, launch_up=False):
    # compute_angle_to_hit_target(hpos[0], hpos[1], hpos[2], bbpos_inner[0], bbpos_inner[1], bbpos_inner[2], bounce_vel)
    # def compute_angle_to_hit_hoop(bbpos_inner, hpos, bbzy, launch_up=True, iv=None):
    # hpos是篮筐的坐标
    # bbpos是篮球击中篮板的坐标
    # iv就是弹出时的速度
    bbpos_inner = np.array(bbpos_inner)
    hpos = np.array(hpos)

    step = math.radians(-0.1)
    initial_angle = math.radians(15)

    if launch_up:
        initial_angle = math.radians(75.0)

    bbxy = bbpos_inner[:2]
    hxy = hpos[:2]
    dx = np.linalg.norm(bbxy - hxy)
    dy = bbpos_inner[2] - hpos[2]

    step_seq = [30, 10, 5, 1, 0.1]

    # can be optimized with newtons or a binary search
    # for now, using simplest method
    # 这里面theta定义很重要，是从哪里开始算零的，代码很模糊
    theta = initial_angle

    it = dx / (iv * math.cos(theta))
    # it是落到篮筐时的时间
    # 感觉应该是dy 减去后面，所以修改了这部分代码
    # py_inner = dy - (math.sin(theta) * it * iv + (G * it ** 2) / 2.0)  #原始代码
    py_inner = dy + (math.sin(theta) * it * iv - (G * it**2) / 2.0)

    above = py_inner > 0
    if not above:
        print("Cannot solve with this sweep")
        return None, None

    last_above = initial_angle
    for s in step_seq:
        above = True
        theta = last_above
        while above:
            theta += step
            it = dx / (iv * math.cos(theta))
            py_inner = dy + (math.sin(theta) * it * iv - (G * it ** 2) / 2.0)
            # py_inner = dy - (math.sin(theta) * it * iv + (G * it ** 2) / 2.0)
            above = py_inner > 0
            if above:
                last_above = theta
    return theta, theta


def compute_angle_to_hit_target(p1x, p1y, p1z, p2x, p2y, p2z, vel):
    # TODO: Filter with max range
    # p1是选手的位置，p2是球击中板的位置，vel是击中板时球速
    dh = p2z - p1z
    p1xy = np.array((p1x, p1y))
    p2xy = np.array((p2x, p2y))
    x_inner = np.linalg.norm(p1xy - p2xy)
    # 不知道原始代码计算出来的夹角究竟是什么角
    sqrt_prod = vel ** 4 - G * (G * x_inner ** 2 + 2 * vel ** 2 * dh)
    if sqrt_prod > 0:
        theta1 = math.atan((vel ** 2 + math.sqrt(sqrt_prod)) / (G * x_inner))
        theta2 = math.atan((vel ** 2 - math.sqrt(sqrt_prod)) / (G * x_inner))

        return theta1, theta2
    return None, None


def p(x_inner, arc):
    v = arc[0]
    t = arc[1]
    return x_inner * math.tan(t) - (G * x_inner ** 2)/(2 * v ** 2 * math.cos(t) ** 2)


def compute_arc_points(arc):
    # v = arc[0]
    # t = arc[1]
    dir_inner = arc[2]
    len_inner = arc[3]
    start = arc[4]

    # return math.sqrt(x)
    pt_count = 300
    sample_pts = np.linspace(0, arc[3], pt_count)
    z_pts = np.array([p(xi, arc) + start[2] for xi in sample_pts])

    xs_inner = (len_inner * dir_inner[0]) / pt_count
    ys_inner = (len_inner * dir_inner[1]) / pt_count

    x_pts = []
    y_pts = []
    px_inner = start[0]
    py_inner = start[1]

    for i in range(pt_count):
        x_pts.append(px_inner)
        y_pts.append(py_inner)
        px_inner += xs_inner
        py_inner += ys_inner

    return x_pts, y_pts, z_pts


def plot_vector(ax_inner, loc, vec, color="blue"):
    nx = [loc[0], loc[0] + vec[0]]
    ny = [loc[1], loc[1] + vec[1]]
    nz = [loc[2], loc[2] + vec[2]]
    ax_inner.plot3D(nx, ny, nz, color)
    # plot_points[0].extend(nx)
    # plot_points[1].extend(ny)
    # plot_points[2].extend(nz)


def draw_circlr_XY(ax_inner, plot_points, center, radius, color='darkred'):
    pt_count = 100
    sample_pts = np.linspace(0, 2.0 * math.pi, pt_count)
    x_pts = np.array([center[0] + math.cos(t) * radius for t in sample_pts])
    y_pts = np.array([center[1] + math.sin(t) * radius for t in sample_pts])
    z_pts = np.empty(pt_count)
    z_pts.fill(center[2])
    ax_inner.plot3D(x_pts, y_pts, z_pts, color)
    plot_points[0].extend(x_pts)
    plot_points[1].extend(y_pts)
    plot_points[2].extend(z_pts)


def draw_rectangle_XZ(ax_inner, plot_points, center, w, h, color='darkred'):
    hw = w * 0.5
    hh = h * 0.5
    cx = center[0]
    cy = center[1]
    cz = center[2]
    nx = [cx-hw, cx+hw, cx+hw, cx-hw, cx-hw]
    ny = [cy, cy, cy, cy, cy]
    nz = [cz-hh, cz-hh, cz+hh, cz+hh, cz-hh]
    ax_inner.plot3D(nx, ny, nz, color)
    plot_points[0].extend(nx)
    plot_points[1].extend(ny)
    plot_points[2].extend(nz)


def plot_arc(ax_inner, plot_points, arc, color='gray'):
    x_line, y_line, z_line = compute_arc_points(arc)
    ax_inner.plot3D(x_line, y_line, z_line, color)

    plot_points[0].extend(x_line)
    plot_points[1].extend(y_line)
    plot_points[2].extend(z_line)


def update_points(num, x0, points, x_moving, y_moving, z_moving, txt):
    txt.set_text('num={:d}'.format(num))  # for debug purposes
    # calculate the new sets of coordinates here. The resulting arrays should have the same shape
    # as the original x,y,z
    # new_x = x+np.random.normal(1,0.1, size=(len(x),))
    # new_y = y+np.random.normal(1,0.1, size=(len(y),))
    # new_z = z+np.random.normal(1,0.1, size=(len(z),))
    new_x = np.reshape(x_moving[num], (len(x0),))
    new_y = np.reshape(y_moving[num], (len(x0),))
    new_z = np.reshape(z_moving[num], (len(x0),))
    # update properties
    points.set_data(new_x, new_y)
    points.set_3d_properties(new_z, 'z')

    # return modified artists
    return points, txt


def plot_moving_arc(fig, ax_inner, plot_points, arc, color='gray'):
    x_line, y_line, z_line = compute_arc_points(arc)
    x_line_arr = np.array(x_line)
    y_line_arr = np.array(y_line)
    z_line_arr = np.array(z_line)
    x0 = np.array([x_line_arr[0]])
    y0 = np.array([y_line_arr[0]])
    z0 = np.array([z_line_arr[0]])
    x_moving = x_line_arr[1:]
    y_moving = y_line_arr[1:]
    z_moving = z_line_arr[1:]
    frame_num = len(z_moving)
    points, = ax_inner.plot(x0, y0, z0, color=color, marker='o')
    txt = fig.suptitle('')
    ax_inner.set_xlabel('X Label')
    ax_inner.set_ylabel('Y Label')
    ax_inner.set_zlabel('Z Label')

    x_scale = np.linspace(-2, 2, 60)
    y_scale = np.linspace(-0.5, 5, 60)
    z_scale = np.linspace(0, 4, 60)
    ax_inner.auto_scale_xyz(x_scale, y_scale, z_scale)
    ani = animation.FuncAnimation(fig, update_points, frames=frame_num, interval=1,
                                  fargs=(x0, points, x_moving, y_moving, z_moving, txt))
    # ax_inner.plot3D(x_line, y_line, z_line, 'r-', alpha=0.7)
    plt.show()
    plot_points[0].extend(x_line)
    plot_points[1].extend(y_line)
    plot_points[2].extend(z_line)


def update_points_twoarcs(num, x0, points, x_moving, y_moving, z_moving, txt, frame_idx):
    txt.set_text('num={:d}'.format(num))  # for debug purposes
    # calculate the new sets of coordinates here. The resulting arrays should have the same shape
    # as the original x,y,z
    # new_x = x+np.random.normal(1,0.1, size=(len(x),))
    # new_y = y+np.random.normal(1,0.1, size=(len(y),))
    # new_z = z+np.random.normal(1,0.1, size=(len(z),))
    new_x = np.reshape(x_moving[frame_idx[num]], (len(x0),))
    new_y = np.reshape(y_moving[frame_idx[num]], (len(x0),))
    new_z = np.reshape(z_moving[frame_idx[num]], (len(x0),))
    # update properties
    points.set_data(new_x, new_y)
    points.set_3d_properties(new_z, 'z')

    # return modified artists
    return points, txt


def plot_moving_arcs(fig, ax_inner, plot_points, arc1, arc2, color='gray'):
    x_line, y_line, z_line = compute_arc_points(arc1)
    x2_line, y2_line, z2_line = compute_arc_points(arc2)

    x_line_arr = np.array(x_line)
    y_line_arr = np.array(y_line)
    z_line_arr = np.array(z_line)
    x2_line_arr = np.array(x2_line)
    y2_line_arr = np.array(y2_line)
    z2_line_arr = np.array(z2_line)

    x_line_arr = np.concatenate((x_line_arr, x2_line_arr))
    y_line_arr = np.concatenate((y_line_arr, y2_line_arr))
    z_line_arr = np.concatenate((z_line_arr, z2_line_arr))

    x0 = np.array([x_line_arr[0]])
    y0 = np.array([y_line_arr[0]])
    z0 = np.array([z_line_arr[0]])

    x_moving = x_line_arr[1:]
    y_moving = y_line_arr[1:]
    z_moving = z_line_arr[1:]

    frame_num = len(z_moving)
    frame_idx = np.linspace(0, frame_num, 50, endpoint=False, dtype=np.int)
    frame_num = frame_idx.shape[0]
    points, = ax_inner.plot(x0, y0, z0, color=color, marker='o', markersize=10)
    txt = fig.suptitle('')
    ax_inner.set_xlabel('X Label')
    ax_inner.set_ylabel('Y Label')
    ax_inner.set_zlabel('Z Label')

    x_scale = np.linspace(-2, 2, 60)
    y_scale = np.linspace(-0.5, 5, 60)
    z_scale = np.linspace(0, 4, 60)
    ax_inner.auto_scale_xyz(x_scale, y_scale, z_scale)

    ani = animation.FuncAnimation(fig, update_points_twoarcs, frames=frame_num, interval=0.1,
                                  fargs=(x0, points, x_moving, y_moving, z_moving, txt, frame_idx))
    ax_inner.plot3D(x_line_arr, y_line_arr, z_line_arr, color='silver')
    # ani.save('./gen_imgs/video2.gif', writer='imagemagick')
    plt.show()
    plot_points[0].extend(x_line)
    plot_points[1].extend(y_line)
    plot_points[2].extend(z_line)


def draw_curves(ax_inner, plot_points, x_curves, xs_inner, ys_inner, offset):
    for y_inner in range(len(x_curves)):
        x_data = []
        y_data = []
        z_data = []

        for x_inner in range(len(x_curves[y_inner])):
            x_data.append(offset[0] + x_inner * xs_inner)
            y_data.append(offset[1] + x_curves[y_inner][x_inner])
            z_data.append(offset[2] + y_inner * ys_inner)

        ax_inner.plot3D(x_data, y_data, z_data, 'darkseagreen')


def mesh_from_normals(w, h, res_x_inner, res_y_inner, desired_normals, draw_y_offset=False):
    # y_offset = normals_to_mesh(avg_normals, res_x, res_y, xs, ys)
    # triangulate a flat mesh
    # perturb heights until normals are as desired
    pts = []
    faces = []
    indexes = []
    normals = []
    touched = []
    ws = w / res_x_inner
    hs = h / res_y_inner

    count_inner = 0
    for y_i in range(res_y_inner):
        index_row = []
        for x_i in range(res_x_inner):
            pt = np.array((x_i * ws, 0.0, y_i * hs))
            pts.append(pt)
            touched.append(False)
            index_row.append(count_inner)
            count_inner += 1
        indexes.append(index_row)

    for y_i in range(res_y_inner - 1):
        for x_i in range(res_x_inner - 1):
            # ip1 = indexes[x_i][y_i]
            # ip2 = indexes[x_i+1][y_i]
            # ip3 = indexes[x_i+1][y_i+1]
            # ip4 = indexes[x_i][y_i+1]
            ip1 = indexes[y_i][x_i]
            ip2 = indexes[y_i][x_i+1]
            ip3 = indexes[y_i+1][x_i+1]
            ip4 = indexes[y_i+1][x_i]
            faces.append([ip1, ip2, ip3])
            faces.append([ip1, ip3, ip4])

            normal1 = desired_normals[x_i][y_i]
            normal2 = desired_normals[x_i][y_i+1]

            normals.append(normal1)
            normals.append(normal2)

    pts = np.array(pts)
    faces = np.array(faces)

    DO_SWEEP = True
    if DO_SWEEP:
        iterations_inner = 1
        for itr_inner in range(iterations_inner):
            # iteratively push triangles to be at normals
            for i in range(len(faces)):
                n = normals[i]
                fp = pts[faces[i][0]]
                # 根据法向量调整的新平面是经过fp点的
                # 所以根据这个点计算出d
                # 再分别计算出其他两个点的y1 y2
                d = np.dot(n, fp)
                p1 = pts[faces[i][1]]
                p2 = pts[faces[i][2]]
                y1 = (d - n[0] * p1[0] - n[2] * p1[2]) / n[1]
                y2 = (d - n[0] * p2[0] - n[2] * p2[2]) / n[1]

                if not touched[faces[i][1]]:
                    pts[faces[i][1]][1] = y1
                    touched[faces[i][1]] = True
                if not touched[faces[i][2]]:
                    pts[faces[i][2]][1] = y2
                    touched[faces[i][2]] = True
    else:
        selections = 500000
        for itr_inner in range(selections):
            faceidx = random.randint(0, len(faces)-1)
            ptidx = random.randint(0, 2)
            n = normals[faceidx]

            fp = pts[faces[faceidx][ptidx]]
            d = np.dot(n, fp)

            p1idx = 1
            p2idx = 2

            if ptidx == 1:
                p1idx = 0
                p2idx = 2
            elif ptidx == 2:
                p1idx = 0
                p2idx = 1

            p1 = pts[faces[faceidx][p1idx]]
            p2 = pts[faces[faceidx][p2idx]]

            y1 = (d - n[0] * p1[0] - n[2] * p1[2]) / n[1]
            y2 = (d - n[0] * p2[0] - n[2] * p2[2]) / n[1]

            pts[faces[faceidx][p1idx]][1] = y1
            pts[faces[faceidx][p2idx]][1] = y2

    model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = pts[f[j], :]

    # Write the mesh to file "cube.stl"
    model.save('test.stl')

    if draw_y_offset:
        # Optionally render the rotated cube faces
        from mpl_toolkits import mplot3d

        # Create a new plot
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)

        # Render the cube
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors))

        # Auto scale to the mesh size
        scale = model.points.flatten('C')
        axes.auto_scale_xyz(scale, scale, scale)

        # # Show the plot to the screen
        plt.show()
        plt.close(figure)

    # y_offset_inner = np.flip(pts[:, 1].reshape((res_y_inner, res_x_inner)))
    y_offset_inner = pts[:, 1].reshape((res_y_inner, res_x_inner))
    return y_offset_inner

