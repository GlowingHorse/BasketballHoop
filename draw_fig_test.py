from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

q = [[-4.32, -2.17, -2.25, 4.72, 2.97, 1.74],
     [ 2.45, 9.73,  7.45,4.01,3.42,  1.80],[-1.40, -1.76, -3.08,-9.94,-3.13,-1.13]]
# v = [[ 0.0068,0.024, -0.014,-0.013, -0.0068,-0.04],[ 0.012,
#       0.056, -0.022,0.016,  0.0045, 0.039],
#      [-0.0045,  0.031,  0.077,0.0016, -0.015,-0.00012]]

x=np.array([0.5])
y=np.array([0.5])
z=np.array([0.5])
x_scale = np.linspace(-10, 10, 21)
x_moving = np.linspace(0, 10, 11)
# s=np.array(v[0])
# u=np.array(v[1])
# w=np.array(v[2])


points, = ax.plot(x, y, z, '*')
txt = fig.suptitle('')


def update_points(num, x, y, z, points, x_moving):
    txt.set_text('num={:d}'.format(num)) # for debug purposes

    # calculate the new sets of coordinates here. The resulting arrays should have the same shape
    # as the original x,y,z
    # new_x = x+np.random.normal(1,0.1, size=(len(x),))
    # new_y = y+np.random.normal(1,0.1, size=(len(y),))
    # new_z = z+np.random.normal(1,0.1, size=(len(z),))
    new_x = x + np.reshape(x_moving[num], (len(x),))
    new_y = y + np.reshape(x_moving[num], (len(y),))
    new_z = z + np.reshape(x_moving[num], (len(y),))
    # update properties
    points.set_data(new_x, new_y)
    points.set_3d_properties(new_z, 'z')

    # return modified artists
    return points, txt


ani = animation.FuncAnimation(fig, update_points, frames=11, fargs=(x, y, z, points, x_moving))
ax.auto_scale_xyz(x_scale, x_scale, x_scale)
plt.show()
plt.close(fig)
