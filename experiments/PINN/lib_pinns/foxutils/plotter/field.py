#usr/bin/python3

#version:0.0.18
#last modified:20240510

from . import *
from .style import *
import collections.abc as collections
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from ..helper.coding import *
from typing import *
import os

def plot3D(z,ztitle="z",xtitle="x",ytitle="y",cmap='viridis',plot2D=False,xlist=None,ylist=None,**kwargs):
    '''
    Plot a 3D surface.

    Args:
        z (torch.Tensor): The input tensor.
        ztitle (str, optional): The title of the z-ax_i. Defaults to "z".
        xtitle (str, optional): The title of the x-ax_i. Defaults to "x".
        ytitle (str, optional): The title of the y-ax_i. Defaults to "y".
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        plot2D (bool, optional): Whether to plot a 2D figure. Defaults to False.
        xlist (list, optional): The list of x-ax_i values. Defaults to None.
        ylist (list, optional): The list of y-ax_i values. Defaults to None.
        kwargs: Additional keyword arguments for the plot_surface.
    '''
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(6, 6))
    delta=1
    if xlist is None:
        xlen=z.shape[0]
        x = np.arange(0, xlen, delta)
    else:
        x=xlist
    if ylist is None:
        ylen=z.shape[1]
        y = np.arange(0, ylen, delta)
    else:
        y=ylist
    Z = z.T
    X, Y = np.meshgrid(x, y)
    surf=ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False,cmap=plt.get_cmap(cmap),**kwargs)  # 设置颜色映射
    plt.xlabel(xtitle,fontsize=12)
    plt.ylabel(ytitle,fontsize=12)
    ax.set_zlabel(ztitle,fontsize=12)
    ax.set_zlim(torch.min(Z).item(), torch.max(Z).item())
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    if plot2D:
        fig, ax = plt.subplots()
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_xticklabels(xlist)
        im=ax.imshow(Z,cmap=plt.get_cmap(cmap))
        plt.colorbar(im,ax=ax)


def plot_2D_ax(ax,
               data,x_start=None,x_end=None,y_start=None,y_end=None,
               transpose=False,
               x_label=None,y_label=None,title=None,title_loc="center",
               interpolation='none', aspect='auto',
               cmap=CMAP_COOLHOT, use_sym_colormap=True,
               show_xy_ticks=True,
               **kwargs):
    """
    Plot a 2D field on the given axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the field.
    - data (numpy.ndarray or torch.Tensor): The 2D field data to be plotted.
    - x_start, x_end, y_start, y_end (float): The range of x and y values for the field.
    - transpose (bool, optional): Whether to transpose the data before plotting. Default is False.
    - x_label, y_label (str, optional): The labels for the x and y axes. Default is None.
    - title (str, optional): The title of the plot. Default is None.
    - title_loc (str, optional): The location of the title. Default is "center".
    - interpolation (str, optional): The interpolation method for the plot. Default is 'none'.
    - aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
    - cmap (matplotlib colormap, optional): The colormap for the plot. Default is CMAP_COOLHOT.
    - sym_colormap (bool, optional): Whether to use a symmetric colormap. Default is True.
    - kwargs: Additional keyword arguments for imshow.

    Returns:
    - im (matplotlib.image.AxesImage): The plotted image.
    """
    x_start=default(x_start,0)
    x_end=default(x_end,data.shape[-2])
    y_start=default(y_start,0)
    y_end=default(y_end,data.shape[-1])
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if transpose:
        data = data.T
        _x_start = y_start;_x_end = y_end
        _y_start = x_start;_y_end = x_end
        _x_label = y_label;_y_label = x_label
    else:
        _x_start = x_start;_x_end = x_end
        _y_start = y_start;_y_end = y_end
        _x_label = x_label;_y_label = y_label
    if use_sym_colormap:
        cmap=sym_colormap(np.min(data), np.max(data), cmap=cmap)
    im=ax.imshow(data, interpolation=interpolation, cmap=cmap, extent=[_x_start, _x_end, _y_start, _y_end],
                  origin='lower', aspect=aspect,**kwargs)
    if not show_xy_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if _x_label is not None:
        ax.set_xlabel(_x_label)
    if _y_label is not None:
        ax.set_ylabel(_y_label)
    if title is not None:
        ax.set_title(title,loc=title_loc)
    return im

def plot_2D(data, x_start=None, x_end=None, y_start=None, y_end=None,
            transpose=False,
            x_label=None, y_label=None, title=None, title_loc="center",
            interpolation='none', aspect='auto',
            cmap=CMAP_COOLHOT, use_sym_colormap=True,
            fig_size=None,
            show_colorbar=True, colorbar_label=None,
            save_path=None,**kwargs):
    """
    Plot a 2D field.

    Parameters:
    - data: 2D array-like object representing the field data.
    - x_start, x_end: Start and end values for the x-ax_i.
    - y_start, y_end: Start and end values for the y-ax_i.
    - transpose: Boolean indicating whether to transpose the data.
    - x_label: Label for the x-ax_i.
    - y_label: Label for the y-ax_i.
    - title: Title of the plot.
    - title_loc: Location of the title ('center', 'left', or 'right').
    - interpolation: Interpolation method for the plot.
    - aspect: Aspect ratio of the plot.
    - cmap: Colormap for the plot.
    - sym_colormap: Boolean indicating whether to use a symmetric colormap.
    - fig_size: Size of the figure (tuple of width and height).
    - show_colorbar: Boolean indicating whether to show the colorbar.
    - colorbar_label: Label for the colorbar.
    - save_path: File path to save the plot.

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    im = plot_2D_ax(ax, data, x_start, x_end, y_start, y_end, transpose=transpose,
                    x_label=x_label, y_label=y_label, title=title, title_loc=title_loc,
                    interpolation=interpolation, cmap=cmap, aspect=aspect,use_sym_colormap=use_sym_colormap,**kwargs)
    if show_colorbar:
        c_bar = fig.colorbar(im)
        if colorbar_label is not None:
            c_bar.set_label(colorbar_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
def plot2D_grid_ax(ax,field,xtitle="i",ytitle="j",cmap='viridis',xlist=None,ylist=None,vmin=None,vmax=None,**kwargs):
    '''
    Generate a 2D plot using matplotlib.pcolormesh().
    
    Args:
        field (torch.Tensor): The input tensor.
        xtitle (str, optional): The title of the x-ax_i. Defaults to "j".
        ytitle (str, optional): The title of the y-ax_i. Defaults to "i".
        cmap (str, optional): The colormap to use. Defaults to 'viridis'.
        xlist (list, optional): The list of x-ax_i values. Defaults to None.
        ylist (list, optional): The list of y-ax_i values. Defaults to None.
        vmin (float, optional): The minimum value of the colormap. Defaults to None.
        vmax (float, optional): The maximum value of the colormap. Defaults to None.
        colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
    '''
    delta=1
    if xlist is None:
        xlen=field.shape[0]
        x = np.arange(0, xlen, delta)
    else:
        x=xlist
    if ylist is None:
        ylen=field.shape[1]
        y = np.arange(0, ylen, delta)
    else:
        y=ylist
    deltax=(x[1]-x[0])/2
    deltay=(y[1]-y[0])/2
    x=np.array([i+deltax for i in x])
    y=np.array([i+deltay for i in y])
    x=np.insert(x,0,x[0]-deltax*2)
    y=np.insert(y,0,y[0]-deltay*2)

    pcm=ax.pcolormesh(x, y, field.T,cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    return pcm
    #fig.set_figheight(len(y)*0.3)
    #fig.set_figwidth(len(x)*0.3)

def plot2D_grid(field,xtitle="i",ytitle="j",cmap='viridis',xlist=None,ylist=None,vmin=None,vmax=None,show_colorbar=True,colorbar_label=None,save_path=None,**kwargs):
    fig, ax = plt.subplots()
    pcm=plot2D_grid_ax(ax,field,xtitle,ytitle,cmap,xlist,ylist,vmin,vmax,**kwargs)
    fig.set_figheight(field.shape[1]*0.3)
    fig.set_figwidth(field.shape[0]*0.3)
    if show_colorbar:
        c_bar=fig.colorbar(pcm)
        if colorbar_label is not None:
            c_bar.set_label(colorbar_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def show_image_from_tensor(image_tensor,title=""):
    '''
    Show an image from a tensor.

    Args:
        image_tensor (torch.Tensor): The input tensor. The shape of the tensor should be (3, height, width).
        title (str, optional): The title of the image. Defaults to "".
    '''
    if len(image_tensor.shape)>3:
        if image_tensor.shape[0]==1:
            image_tensor=image_tensor.squeeze(dim=0)
        else:
            print("wrong input type")
    n_channels = image_tensor.shape[-3]
    for c in range(n_channels):
        maxv=torch.max(image_tensor[c])
        minv=torch.min(image_tensor[c])
        image_tensor[c] = (image_tensor[c]-minv)/(maxv-minv)
    plt.title(title,y=-0.1)
    plt.imshow(image_tensor.permute(1,2,0))
    plt.ax_i('off')

class ChannelPloter():
    """
    A class for plotting channel fields.

    Methods:
    - fig_save_path(self, path): Sets the figure save path.
    - plot(self, fields, channel_names, channel_units, case_names, title, transpose, inverse_y, cmap, mask, size_subfig, xspace, yspace, cbar_pad, title_position, redraw_cticks, num_colorbar_value, minvs, maxvs, ctick_format, data_scale, rotate_colorbar_with_oneinput, subfigure_index, save_name, use_sym_colormap): Plots the fields.
    """

    def __type_transform(self, fields):
        """
        Transforms the input fields to the desired type.

        Args:
        - fields: The input fields.

        Returns:
        - The transformed fields.
        """
        if isinstance(fields, collections.Sequence):
            if isinstance(fields[0], torch.Tensor):
                fields = [(field.to(torch.device("cpu"))).numpy() for field in fields]
                return fields
            elif isinstance(fields[0], np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")
        else:
            if isinstance(fields, torch.Tensor):
                fields = (fields.to(torch.device("cpu"))).numpy()
                return fields
            elif isinstance(fields, np.ndarray):
                return fields
            else:
                raise Exception("Wrong input type!")       
        
    def __cat_fields(self, fields):
        """
        Concatenates the fields into a single array.

        Args:
        - fields: The input fields.

        Returns:
        - The concatenated fields.
        """
        if isinstance(fields, collections.Sequence):
            if len(fields[0].shape) == 4:
                return np.concatenate(fields, 0)
            elif len(fields[0].shape) == 3:
                return np.concatenate([np.expand_dims(field, 0) for field in fields], 0)
            elif len(fields[0].shape) == 2:
                return np.concatenate([np.expand_dims(np.expand_dims(field, 0), 0) for field in fields], 0)
            else:
                raise Exception("Wrong input type!")
        else:
            if len(fields.shape) == 2:
                return np.expand_dims(np.expand_dims(fields, 0), 0)
            if len(fields.shape) == 3:
                return np.expand_dims(fields, 0)
            elif len(fields.shape) == 4:
                return fields
            else:
                raise Exception("Wrong input type!")

    def __find_min_max(self, fields, defaultmin, defaultmax):
        """
        Finds the minimum and maximum values for each field.

        Args:
        - fields: The input fields.
        - defaultmin: The default minimum values.
        - defaultmax: The default maximum values.

        Returns:
        - The minimum and maximum values for each field.
        """
        mins = []
        maxs = []
        for i in range(fields.shape[1]):
            if defaultmin is not None:
                if defaultmin[i] is not None:
                    mins.append(defaultmin[i])
                else:
                    mins.append(np.min(fields[:, i, :, :]))
            else:
                mins.append(np.min(fields[:, i, :, :]))
            if defaultmax is not None:
                if defaultmax[i] is not None:
                    maxs.append(defaultmax[i])
                else:
                    maxs.append(np.max(fields[:, i, :, :]))
            else:
                maxs.append(np.max(fields[:, i, :, :]))
        return mins, maxs
   
    def __generate_mask(self, mask, transpose, color="white"):
        """
        Generates a mask for the fields.

        Args:
        - mask: The mask.
        - transpose: Whether to transpose the mask.
        - color: The color of the mask.

        Returns:
        - The generated mask.
        """
        mask = self.__type_transform(mask)
        if color == "white":
            RGB = np.ones(mask.shape)  # zeros=Black, ones=white
        elif color == "black":   
            RGB = np.zeros(mask.shape) 
        if transpose:
            return torch.cat([np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(mask.T, 2)], -1)
        else:
            return torch.cat([np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(RGB, 2), np.expand_dims(mask, 2)], -1)
        
    def plot(self, fields: torch.Tensor|np.ndarray|Sequence,
                 channel_names:Optional[Sequence]=None, channel_units:Optional[Sequence]=None, 
                 case_names:Optional[Sequence]=None, 
                 title:str="", title_position:float=0.0,
                 transpose:bool=False, inverse_y:bool=False,aspect='auto',
                 data_scale:Optional[Sequence]=None, mask=None, 
                 size_subfig:float=3.5, xspace:float=0.7, yspace:float=0.1, 
                 x_start:Optional[float]=None, x_end:Optional[float]=None, y_start:Optional[float]=None, y_end:Optional[float]=None,
                 minvs:Optional[Sequence]=None, maxvs:Optional[Sequence]=None,
                 cmap=CMAP_COOLHOT, use_sym_colormap:bool=True,
                 cbar_pad:float=0.1, redraw_cticks:bool=True, num_colorbar_value:int=4, ctick_format:Optional[str]=None, 
                 rotate_colorbar_with_oneinput:bool=False, 
                 subfigure_index:Optional[Sequence]=None, 
                 save_name:Optional[str]=None, 
                 show_x_y_ticks:bool=False):
            """
            Plot the fields.

            Args:
                fields (torch.Tensor|np.ndarray|Sequence): The fields to be plotted.
                channel_names (Optional[Sequence], optional): The names of the channels. Defaults to None.
                channel_units (Optional[Sequence], optional): The units of the channels. Defaults to None.
                case_names (Optional[Sequence], optional): The names of the cases. Defaults to None.
                title (str, optional): The title of the plot. Defaults to "".
                title_position (float, optional): The position of the title. Defaults to 0.0.
                transpose (bool, optional): Whether to transpose the fields. Defaults to False.
                inverse_y (bool, optional): Whether to invert the y-axis. Defaults to False.
                aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
                data_scale (Optional[Sequence], optional): The scale of the data. Defaults to None.
                mask (optional): The mask to be applied to the plot. Defaults to None.
                size_subfig (float, optional): The size of the subfigure. Defaults to 3.5.
                xspace (float, optional): The space between x-axis labels. Defaults to 0.7.
                yspace (float, optional): The space between y-axis labels. Defaults to 0.1.
                x_start (Optional[float], optional): The start value of the x-axis. Defaults to None.
                x_end (Optional[float], optional): The end value of the x-axis. Defaults to None.
                y_start (Optional[float], optional): The start value of the y-axis. Defaults to None.
                y_end (Optional[float], optional): The end value of the y-axis. Defaults to None.
                minvs (Optional[Sequence], optional): The minimum values for the colorbar. Defaults to None.
                maxvs (Optional[Sequence], optional): The maximum values for the colorbar. Defaults to None.
                cmap (optional): The colormap to be used. Defaults to CMAP_COOLHOT.
                use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to True.
                cbar_pad (float, optional): The padding of the colorbar. Defaults to 0.1.
                redraw_cticks (bool, optional): Whether to redraw the colorbar ticks. Defaults to True.
                num_colorbar_value (int, optional): The number of colorbar values. Defaults to 4.
                ctick_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
                rotate_colorbar_with_oneinput (bool, optional): Whether to rotate the colorbar with one input. Defaults to False.
                subfigure_index (Optional[Sequence], optional): The index of the subfigure. Defaults to None.
                save_name (Optional[str], optional): The name of the saved plot. Defaults to None.
                show_x_y_ticks (bool, optional): Whether to show x and y ticks. Defaults to False.
            """
            fields = self.__cat_fields(self.__type_transform(fields))
            if mask is not None:
                mask = self.__generate_mask(mask, transpose=transpose)
            num_cases = fields.shape[0]
            num_channels = fields.shape[1]
            
            channel_names = default(channel_names, ["channel {}".format(i) for i in range(num_channels)])
            channel_units = default(channel_units, ["" for i in range(num_channels)])
            case_names = default(case_names, ["case {}".format(i) for i in range(num_cases)])
            data_scale = default(data_scale, [1 for i in range(num_channels)])
            fields = np.concatenate([fields[:, i:i+1, :, :] * data_scale[i] for i in range(num_channels)], 1)
            mins, maxs = self.__find_min_max(fields, minvs, maxvs)
            
            if num_cases == 1 and rotate_colorbar_with_oneinput:
                cbar_location = "right"
                cbar_mode = 'each'
                ticklocation = "right"
            else:
                cbar_location = "top"
                cbar_mode = 'edge'
                ticklocation = "top" 
            
            fig = plt.figure(figsize=(size_subfig * num_channels, size_subfig * num_cases))
            grid = ImageGrid(fig, 111,
                            nrows_ncols=(num_cases, num_channels),
                            axes_pad=(xspace, yspace),
                            share_all=True,
                            cbar_location=cbar_location,
                            cbar_mode=cbar_mode,
                            direction='row',
                            cbar_pad=cbar_pad
                            )
            im_cb = []
              
            for i, ax_i in enumerate(grid):
                i_row = i // num_channels
                i_column = i % num_channels
                datai = fields[i_row, i_column]
                if i_column == 0:
                    y_label=case_names[i_row]
                else:
                    y_label=None
                if i_row == num_cases - 1:
                    x_label=channel_names[i_column]  
                else:
                    x_label=None            
                im=plot_2D_ax(
                    ax=ax_i,data=datai,
                    x_start=x_start,x_end=x_end,y_start=y_start,y_end=y_end,
                    transpose=transpose,
                    use_sym_colormap=use_sym_colormap,
                    show_xy_ticks=show_x_y_ticks,
                    x_label=x_label,y_label=y_label,
                    cmap=cmap,
                    aspect=aspect,
                )
                if i < num_channels:
                    im_cb.append(im)      
                if mask is not None:
                    ax_i.imshow(mask)  
                if inverse_y:
                    ax_i.invert_yax_i() 

            for i in range(num_channels):
                cb = grid.cbar_axes[i].colorbar(im_cb[i], label=channel_units[i], ticklocation=ticklocation, format=ctick_format)
                cb.ax.minorticks_on()
                if redraw_cticks:
                    cb.set_ticks(np.linspace(mins[i], maxs[i], num_colorbar_value, endpoint=True))      
            fig.suptitle(title, y=title_position)
            if subfigure_index is not None:
                plt.suptitle(subfigure_index, x=0.01, y=0.88, fontproperties="Times New Roman")
            if save_name is not None:
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                plt.savefig(save_name, bbox_inches='tight')
            plt.show()

channel_plotter=ChannelPloter()

def show_each_channel(fields: torch.Tensor|np.ndarray|Sequence,
                 channel_names:Optional[Sequence]=None, channel_units:Optional[Sequence]=None, 
                 case_names:Optional[Sequence]=None, 
                 title:str="", title_position:float=0.0,
                 transpose:bool=False, inverse_y:bool=False, aspect="auto",
                 data_scale:Optional[Sequence]=None, mask=None, 
                 size_subfig:float=3.5, xspace:float=0.7, yspace:float=0.1, 
                 x_start:Optional[float]=None, x_end:Optional[float]=None, y_start:Optional[float]=None, y_end:Optional[float]=None,
                 minvs:Optional[Sequence]=None, maxvs:Optional[Sequence]=None,
                 cmap=CMAP_COOLHOT, use_sym_colormap:bool=True,
                 cbar_pad:float=0.1, redraw_cticks:bool=True, num_colorbar_value:int=4, ctick_format:Optional[str]=None, 
                 rotate_colorbar_with_oneinput:bool=False, 
                 subfigure_index:Optional[Sequence]=None, 
                 save_name:Optional[str]=None, 
                 show_x_y_ticks:bool=False):
    """
    Plot the fields.

    Args:
        fields (torch.Tensor|np.ndarray|Sequence): The fields to be plotted.
        channel_names (Optional[Sequence], optional): The names of the channels. Defaults to None.
        channel_units (Optional[Sequence], optional): The units of the channels. Defaults to None.
        case_names (Optional[Sequence], optional): The names of the cases. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "".
        title_position (float, optional): The position of the title. Defaults to 0.0.
        transpose (bool, optional): Whether to transpose the fields. Defaults to False.
        inverse_y (bool, optional): Whether to invert the y-axis. Defaults to False.
        aspect (str, optional): The aspect ratio of the plot. Default is 'auto'.
        data_scale (Optional[Sequence], optional): The scale of the data. Defaults to None.
        mask (optional): The mask to be applied to the plot. Defaults to None.
        size_subfig (float, optional): The size of the subfigure. Defaults to 3.5.
        xspace (float, optional): The space between x-axis labels. Defaults to 0.7.
        yspace (float, optional): The space between y-axis labels. Defaults to 0.1.
        x_start (Optional[float], optional): The start value of the x-axis. Defaults to None.
        x_end (Optional[float], optional): The end value of the x-axis. Defaults to None.
        y_start (Optional[float], optional): The start value of the y-axis. Defaults to None.
        y_end (Optional[float], optional): The end value of the y-axis. Defaults to None.
        minvs (Optional[Sequence], optional): The minimum values for the colorbar. Defaults to None.
        maxvs (Optional[Sequence], optional): The maximum values for the colorbar. Defaults to None.
        cmap (optional): The colormap to be used. Defaults to CMAP_COOLHOT.
        use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to True.
        cbar_pad (float, optional): The padding of the colorbar. Defaults to 0.1.
        redraw_cticks (bool, optional): Whether to redraw the colorbar ticks. Defaults to True.
        num_colorbar_value (int, optional): The number of colorbar values. Defaults to 4.
        ctick_format (Optional[str], optional): The format of the colorbar ticks. Defaults to None.
        rotate_colorbar_with_oneinput (bool, optional): Whether to rotate the colorbar with one input. Defaults to False.
        subfigure_index (Optional[Sequence], optional): The index of the subfigure. Defaults to None.
        save_name (Optional[str], optional): The name of the saved plot. Defaults to None.
        show_x_y_ticks (bool, optional): Whether to show x and y ticks. Defaults to False.
    """
    channel_plotter.plot(
        fields=fields,
        channel_names=channel_names, channel_units=channel_units,
        case_names=case_names,
        title=title, title_position=title_position,
        transpose=transpose, inverse_y=inverse_y,aspect=aspect,
        data_scale=data_scale, mask=mask,
        size_subfig=size_subfig, xspace=xspace, yspace=yspace,
        x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end,
        minvs=minvs, maxvs=maxvs,
        cmap=cmap, use_sym_colormap=use_sym_colormap,
        cbar_pad=cbar_pad, redraw_cticks=redraw_cticks, num_colorbar_value=num_colorbar_value, ctick_format=ctick_format,
        rotate_colorbar_with_oneinput=rotate_colorbar_with_oneinput,
        subfigure_index=subfigure_index,
        save_name=save_name,
        show_x_y_ticks=show_x_y_ticks
    )