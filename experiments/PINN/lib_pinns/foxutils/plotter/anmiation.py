#usr/bin/python3

#version:0.0.2
#last modified:20231210

from . import *
import torch
from matplotlib import animation

def save_animations_from_tensor(tensors, save_path, Transpose=True, inverse_y=True, cmap='viridis', interval=50, timefunc=None):
    """
    Save animations from a tensor to a gif file.

    Args:
        tensors (torch.Tensor): The tensor containing the frames of the animation. The shape of the tensor should be (n_frames, height, width).
        save_path (str): The path to save the animation file.
        Transpose (bool, optional): Whether to transpose the tensor. Defaults to True.
        inverse_y (bool, optional): Whether to invert the y-axis. Defaults to True.
        cmap (str, optional): The colormap to use for the animation. Defaults to 'viridis'.
        interval (int, optional): The interval between frames in milliseconds. Defaults to 50.
        timefunc (function, optional): A function to generate the time label for each frame. Defaults to None.
    """
    vmin = torch.min(tensors)
    vmax = torch.max(tensors)
    if Transpose:
        tensors = tensors.transpose(1, 2)
    fig, ax = plt.subplots()

    def animate(i):
        ax.cla()
        ax.set_axis_off()
        frame = ax.imshow(tensors[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        if timefunc is not None:
            ax.set_title("$t=${}".format(timefunc(i)))
        if inverse_y:
            ax.invert_yaxis()
        return frame,

    animation1 = animation.FuncAnimation(fig=fig, func=animate, frames=tensors.shape[0], interval=interval, blit=False)
    animation1.save(save_path, writer='imagemagick')
