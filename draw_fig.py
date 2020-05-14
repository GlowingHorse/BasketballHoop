import numpy as np
import math
import matplotlib.pyplot as plt
import random
from stl import mesh
import pickle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from utils import *

G = 9.8  # m/s^2，不知道单位对不对
BACKBOARD_HEIGHT_ABOVE_GROUND_M = 3
BACKBOARD_HEIGHT_M = 0.6
BACKBOARD_WIDTH_M = 0.8
HOOP_DIAMETER_M = 0.4

HALF_WIDTH = BACKBOARD_WIDTH_M / 2

PLAYER_HEIGHT_M = 1.8
ELASTICITY_COFFICIENT = 0.8

# 1 miles per hour = 0.44704 meter per second
# For example, when shooting a 2-foot shot, you only need
# a launch speed of approximately 10 miles per hour.
# For a 3-point shot you need a launch speed of approximately 18 miles per hour.
MAX_SHOT_VELOCITY_MPS = 8
MIN_SHOT_VELOCITY_MPS = 5.8

# 篮板的x方向和y方向（实际后面的z方向）的分辨率
# res_x = 80的话等于将真实篮板的宽度分成了80份
# y（实际空间的z）同理，对应的是篮板空间的高度
# 在计算实际空间的y，也就球落到篮筐的y时，是直接读取的数据
res_x = 32
res_y = 24

# 把板按比例分好，然后给随机弹出速度和全部的位置。
# 计算落到篮筐时的x=0情况下全部可能的y，把最可能的y记录到对应的内容里

xs = 0.8 / res_x
ys = 0.6 / res_y

max_angle = 1.57
min_angle = -1.57


def draw_components(p_loc, hit_loc, hoop_loc, arc_1, arc_2, normal, ax_inner, draw_clear=True,
                    fig_inner=None, draw_moving=False):
    # Player hit hoop loc in X, Y, Z
    # arc is velocity, Z angle, XY direction, length, start, bounce_normal

    plot_points = [[], [], []]

    if ax_inner is None:
        fig_inner = plt.figure()
        ax_inner = fig_inner.add_subplot(111, projection='3d')

    # x_data = [p_loc[0], hit_loc[0], hoop_loc[0]]
    # y_data = [p_loc[1], hit_loc[1], hoop_loc[1]]
    # z_data = [p_loc[2], hit_loc[2], hoop_loc[2]]

    x_data = [0]
    y_data = [0]
    z_data = [0]

    # ax_inner.scatter3D(x_data, y_data, z_data, cmap='Greens')
    plot_points[0].extend(x_data)
    plot_points[1].extend(y_data)
    plot_points[2].extend(z_data)

    inbound_norm = arc_1[5]
    outbound_norm = arc_2[5]

    if not draw_clear:
        plot_vector(ax_inner, hit_loc, normal, "cyan")
        plot_vector(ax_inner, hit_loc, inbound_norm * 3, "black")
        plot_vector(ax_inner, hit_loc, outbound_norm * 3, "white")

    draw_rectangle_XZ(ax_inner, plot_points,
                      (0, 0, BACKBOARD_HEIGHT_ABOVE_GROUND_M + BACKBOARD_HEIGHT_M * 0.5),
                      BACKBOARD_WIDTH_M, BACKBOARD_HEIGHT_M, color='darkolivegreen')
    draw_circlr_XY(ax_inner, plot_points, (0, HOOP_DIAMETER_M * 0.5, BACKBOARD_HEIGHT_ABOVE_GROUND_M),
                   HOOP_DIAMETER_M * 0.5, color='darkolivegreen')

    if not draw_clear:
        plot_arc(ax_inner, plot_points, arc_1, 'lightgray')
        plot_arc(ax_inner, plot_points, arc_2, 'yellow')
    else:
        if draw_moving:
            # plot_moving_arc(fig_inner, ax_inner, plot_points, arc_1, 'lightsteelblue')
            # plot_moving_arc(fig_inner, ax_inner, plot_points, arc_2, 'blue')
            plot_moving_arcs(fig_inner, ax_inner, plot_points, arc_1, arc_2, 'saddlebrown')
        else:
            plot_arc(ax_inner, plot_points, arc_1, 'lightsteelblue')
            plot_arc(ax_inner, plot_points, arc_2, 'blue')

    # draw_curves(ax_inner, y_offset, xs, ys, [-HALF_WIDTH, 0, BACKBOARD_HEIGHT_ABOVE_GROUND_M])
    FORCE_SQUARE = False
    if FORCE_SQUARE:
        X = np.array(plot_points[0])
        Y = np.array(plot_points[1])
        Z = np.array(plot_points[2])

        max_range = np.max(np.array((np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z))))
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (np.max(X) + np.min(X))
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (np.max(Y) + np.min(Y))
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (np.max(Z) + np.min(Z))
        # max_range = np.array((X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())).max()
        # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncommnet following both lines to test the fake bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax_inner.plot([xb], [yb], [zb], 'w')

    plt.show()


def xy_to_bb_pos(x_inner, y_inner, y_offset_inner):
    # 篮板的x方向和y方向（实际后面的z方向）的分辨率
    # res_x = 50的话等于将真实篮板的宽度分成了50份
    # y（实际空间的z）同理，对应的是篮板空间的高度
    # 在计算实际空间的y，也就球落到篮筐的y时，是直接读取的数据
    x_inner += 1
    y_inner += 1
    xs_inner = BACKBOARD_WIDTH_M / res_x
    ys_inner = BACKBOARD_HEIGHT_M / res_y
    hx_inner = -HALF_WIDTH + xs_inner * x_inner
    hrx = res_x * 0.5

    if x_inner >= hrx:
        x_inner = int(hrx - (x_inner - hrx) - 1)

    hy_inner = y_offset_inner[y_inner][x_inner]  # 原始代码是先y再x
    # hy_inner = y_offset[x_inner][y_inner]
    hz_inner = BACKBOARD_HEIGHT_ABOVE_GROUND_M + ys_inner * y_inner
    return np.array((hx_inner, hy_inner, hz_inner))


def compute_normal_for_bounce(ploc_inner, bbpos_inner, vmin, vmax, hpos, iterations_inner=100,
                              draw_inner=False, ax_inner=None, draw_clear=False
                              , fig_inner=None, draw_moving=True):
    iteration_per_cell = iterations_inner
    iter_count = 0

    reflection_sum = np.zeros(3)

    while iter_count < iteration_per_cell:
        vel = random.uniform(vmin, vmax)
        a1, a2 = compute_angle_to_hit_target(ploc_inner[0], ploc_inner[1], PLAYER_HEIGHT_M, bbpos_inner[0], bbpos_inner[1], bbpos_inner[2], vel)

        if a1 is not None and a2 is not None:
            inbound_angle = None

            # Hack to filter out shots that are super high angle or line drive
            if min_angle <= a2 <= max_angle:
                inbound_angle = a2

            if inbound_angle is None:
                continue

            # First the inbound shot
            # for now targeting only the center of the hoop but will expand to capture most valid hoop locations
            # 篮筐x坐标为0，y坐标为半径，z坐标为距地面高度
            hoopx = 0.0  # + BACKBOARD_WIDTH_M
            hoopy = HOOP_DIAMETER_M * 0.5  # - HOOP_DIAMETER_M * 0.7 * 0.5 # HOOP_DIAMETER_M * 0.5 # * 0.1
            hoopz = BACKBOARD_HEIGHT_ABOVE_GROUND_M

            pv = np.array(ploc_inner[:2])  # XY location of player
            hv = np.array(bbpos_inner[:2])  # XY location of ball bounce on hoop
            bv = np.array((hoopx, hoopy))  # XY location of basket center

            player_to_backboard_dist_XY = np.linalg.norm(hv - pv)
            dz_inbound = math.tan(inbound_angle) - (G * player_to_backboard_dist_XY) / (
                math.cos(inbound_angle) ** 2 * vel ** 2)

            # dz is the slope with respect to 1 unit of travel in XY so we can take X+Y components and normalize
            # Then form a vector with dz and normalize again and we have a normal representing the approach vector
            player_to_bb_XY = pv - hv
            player_to_bb_XY_norm = player_to_bb_XY / np.linalg.norm(player_to_bb_XY)
            inbound_vector = np.append(-player_to_bb_XY_norm, dz_inbound)
            # 加入入射时是抛物线的后半部分，那么入射速度对应的法向量的xyz三个坐标应该都是负
            inbound_norm = -(inbound_vector / np.linalg.norm(inbound_vector))

            # Compute the inbound velocity
            # Doing some annoying math to get velocity of X & Y components of motion
            # 入射速度乘
            x_vel = vel * math.cos(inbound_angle)
            y_vel = x_vel * 1.0 * abs(dz_inbound)
            inbound_speed = np.linalg.norm(np.array((x_vel, y_vel)))

            # Elasticity determines speed after bounce
            bounce_vel = inbound_speed * ELASTICITY_COFFICIENT

            # Surprisingly the math is essentially the same
            # There is a bounce at some speed and we need to compute the launch angle to go into the hoop

            MAKE_LOWER_EDGES = False
            if MAKE_LOWER_EDGES:
                if (bbpos_inner[0] < -.15 or bbpos_inner[0] > 0.15) and (bbpos_inner[2] - BACKBOARD_HEIGHT_ABOVE_GROUND_M) < 0.15:
                    rebound_angle1, rebound_angle2 = compute_angle_to_hit_target(bbpos_inner[0], bbpos_inner[1], bbpos_inner[2],
                                                                                 hoopx, hoopy, hoopz, bounce_vel)
                else:
                    rebound_angle1, rebound_angle2 = compute_angle_to_hit_hoop(bbpos_inner, (hoopx, hoopy, hoopz), bounce_vel)
            else:
                rebound_angle1, rebound_angle2 = compute_angle_to_hit_hoop(bbpos_inner, (hoopx, hoopy, hoopz), bounce_vel, launch_up=True)

            if rebound_angle1 is not None and rebound_angle2 is not None:
                outbound_angle = max(rebound_angle2, rebound_angle1)
                if hoopz - BACKBOARD_HEIGHT_ABOVE_GROUND_M < 0.2:
                    outbound_angle = rebound_angle1

                # location on backboard
                bx = int(math.floor(HALF_WIDTH + bbpos_inner[0]) / BACKBOARD_WIDTH_M * res_x)
                bz = int(math.floor((bbpos_inner[2] - BACKBOARD_HEIGHT_ABOVE_GROUND_M) / BACKBOARD_HEIGHT_M * res_y))

                # Method 2 - we will compute the specific normal to achieve the desired bounce
                # This will be done by computing the bisector of the inbound vector and the desired outbound vector in 3 space
                # Getting the inbound and outbound vectors will be desired using the derivative of the parabolic trajectories
                # There are also several combinations possible with two launch angles and two rebound angles
                # We will for now just angle I for both options

                # Now the outbound shot
                backboard_to_hoop_dist_XY = np.linalg.norm(bv - hv)
                # 不知道是不是desired z_outbound的意思
                dz_outbound = math.tan(outbound_angle) - (G * 0.0) / (math.cos(outbound_angle) ** 2 * bounce_vel ** 2)

                # Same deal as inbound
                # bv是篮筐中心，hv是击中篮板的点
                hoop_to_bb = bv - hv
                hoop_to_bb_XY_norm = hoop_to_bb / np.linalg.norm(hoop_to_bb)
                outbound_vector = np.append(hoop_to_bb_XY_norm, dz_outbound)
                outbound_norm = outbound_vector / np.linalg.norm(outbound_vector)

                # The goal normal is just the normalized sum of the two vectors
                bisector_vector = inbound_norm + outbound_norm
                reflection_normal = bisector_vector / np.linalg.norm(bisector_vector)
                if draw_inner:
                    p_loc = np.array((ploc_inner[0], ploc_inner[1], PLAYER_HEIGHT_M))
                    hit_loc = np.array(hpos)
                    hoop_loc = np.array((hoopx, hoopy, hoopz))
                    normal = reflection_normal
                    shot_vector = hv - pv
                    # compute np.linalg.norm 2-norm
                    shot_length_XY = np.linalg.norm(shot_vector)
                    shot_direction_XY = shot_vector / shot_length_XY
                    bounce_vector_XY = hoop_to_bb
                    bounce_length_XY = np.linalg.norm(bounce_vector_XY)
                    bounce_direction_XY = bounce_vector_XY / bounce_length_XY

                    arc_1 = [vel, inbound_angle, shot_direction_XY, shot_length_XY, p_loc, inbound_norm]
                    arc_2 = [bounce_vel, outbound_angle, bounce_direction_XY, bounce_length_XY, hit_loc, outbound_norm]
                    # arc_2 = [vel, a2, shot_direction_XY, shot_length_XY, p_loc]

                    draw_components(p_loc, hit_loc, hoop_loc, arc_1, arc_2, normal, ax_inner,
                                    draw_clear=draw_clear, fig_inner=fig_inner, draw_moving=draw_moving)

                reflection_sum += reflection_normal

                # count[bx][bz] += 1

                iter_count += 1
            else:
                pass
                # print("No rebound possible")
    reflection_sum /= iteration_per_cell
    return reflection_sum


SAVE_NORMALS = False
train_flag = True

if not train_flag:
    ploc = [0.0, 2.5, PLAYER_HEIGHT_M]

    # count = np.zeros((BACKBOARD_WIDTH_M, BACKBOARD_HEIGHT_M))
    # avg_normals = np.zeros((res_x, res_y))

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    ax.view_init(azim=135)
    y_offset = pickle.load(open("./models/last_y_off.p", "rb"))
    # y_offset = np.zeros((res_y, res_x))
    y_offset -= np.min(y_offset)
    draw_curves(ax, [], y_offset, xs, ys, [-HALF_WIDTH, 0, BACKBOARD_HEIGHT_ABOVE_GROUND_M])

    for x in range(0, res_x, 9):
        for y in range(9, res_y, 8):
            bbpos = xy_to_bb_pos(x, y, y_offset)
            hx = bbpos[0]
            hy = bbpos[1]
            hz = bbpos[2]
            compute_normal_for_bounce(ploc, bbpos, MIN_SHOT_VELOCITY_MPS, MAX_SHOT_VELOCITY_MPS,
                                      (hx, hy, hz), iterations_inner=1,
                                      draw_inner=True, ax_inner=ax, draw_clear=True)
    # x = 7
    # y = 5
    #
    # bbpos = xy_to_bb_pos(x, y, y_offset)
    # hx = bbpos[0]
    # hy = bbpos[1]
    # hz = bbpos[2]
    # compute_normal_for_bounce(ploc, bbpos, MIN_SHOT_VELOCITY_MPS, MAX_SHOT_VELOCITY_MPS, (hx, hy, hz), iterations_inner=1,
    #                           draw_inner=True, ax_inner=ax, draw_clear=True)
    plt.show()

elif train_flag:
    y_offset = pickle.load(open("./models/last_y_off.p", "rb"))
    avg_normals = pickle.load(open("./models/avg_norms.p", "rb"))
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    ax.view_init(elev=20, azim=5)
    draw = True
    bbpos = xy_to_bb_pos(16, 12, y_offset)
    hx = bbpos[0]
    hy = bbpos[1]
    hz = bbpos[2]
    draw_curves(ax, [], y_offset, xs, ys, [-HALF_WIDTH, 0, BACKBOARD_HEIGHT_ABOVE_GROUND_M])
    bounce_normal = compute_normal_for_bounce((0, 2.5, PLAYER_HEIGHT_M), bbpos,
                                              MIN_SHOT_VELOCITY_MPS,
                                              MAX_SHOT_VELOCITY_MPS, (hx, hy, hz), iterations_inner=10,
                                              draw_inner=draw, ax_inner=ax, draw_clear=True
                                              , fig_inner=fig, draw_moving=True)
    y_offset = mesh_from_normals(BACKBOARD_WIDTH_M, BACKBOARD_HEIGHT_M, res_x, res_y, avg_normals, draw_y_offset=True)

    # check whether right
    avg_normals = avg_normals / np.linalg.norm(avg_normals, axis=-1)
    if SAVE_NORMALS:
        avg_x = []
        avg_y = []
        avg_z = []
        for y in range(res_y):
            for x in range(res_x):
                bbpos = xy_to_bb_pos(x, y, y_offset)
                norm = avg_normals[x][y]
                avg_x.append((norm[0] + 1.0) * 0.5)
                avg_y.append((norm[1] + 1.0) * 0.5)
                avg_z.append((norm[2] + 1.0) * 0.5)

                # X.append(bbpos[0])
                # Y.append(bbpos[1])
                # Z.append(bbpos[2])
                # U.append(norm[0])
                # V.append(norm[1])
                # W.append(norm[2])

        print(len(avg_x))

        from PIL import Image
        rgbArray = np.zeros((res_x, res_y, 3), np.int8)
        rgbArray[..., 0] = np.array(avg_x).reshape((res_y, res_x)) * 256
        rgbArray[..., 1] = np.array(avg_y).reshape((res_y, res_x)) * 256
        rgbArray[..., 2] = np.array(avg_z).reshape((res_y, res_x)) * 256
        img = Image.fromarray(rgbArray)
        img.save('./gen_imgs/normals.png')
