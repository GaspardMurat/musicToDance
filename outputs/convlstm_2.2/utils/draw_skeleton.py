import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

path_to_skeleton = 'dataset_master/DANCE_C_1'
chosen_frame = 0

with open('../%s/skeletons.json' % path_to_skeleton, 'r') as fin:
    skeletons = np.array(json.load(fin)['skeletons'])


def draw_skeleton(data, frame):
    skeleton = data[frame]
    pts = list()
    pts.append(skeleton[0, :])  # head-left
    pts.append(skeleton[1, :])  # head-right
    pts.append(skeleton[2, :])  # Waist
    pts.append(skeleton[3, :])  # Shoulder-left
    pts.append(skeleton[4, :])  # Elbow-left
    pts.append(skeleton[5, :])  # Wrist-left
    pts.append(skeleton[6, :])  # Hand-left
    pts.append(skeleton[7, :])  # Waist-left
    pts.append(skeleton[8, :])  # Knee-left
    pts.append(skeleton[9, :])  # Ankle-left
    pts.append(skeleton[10, :])  # Heel-left
    pts.append(skeleton[11, :])  # Toe-left
    pts.append(skeleton[12, :])  # Shoulder-right
    pts.append(skeleton[13, :])  # Elbow-right
    pts.append(skeleton[14, :])  # Wrist-right
    pts.append(skeleton[15, :])  # Hand-right
    pts.append(skeleton[16, :])  # Waist-right
    pts.append(skeleton[17, :])  # Knee-right
    pts.append(skeleton[18, :])  # Ankle-right
    pts.append(skeleton[19, :])  # Heel-right
    pts.append(skeleton[20, :])  # Toe-right
    pts.append(skeleton[21, :])  # For sampling 1 (left)
    pts.append(skeleton[22, :])  # For sampling 2 (right)

    x_axis = list()
    y_axis = list()
    z_axis = list()
    for i in range(len(pts)):
        x_axis.append(pts[i][0])
        y_axis.append(pts[i][1])
        z_axis.append(pts[i][2])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_axis, z_axis, y_axis, marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    plt.show()


if __name__ == '__main__':
    draw_skeleton(skeletons, chosen_frame)
