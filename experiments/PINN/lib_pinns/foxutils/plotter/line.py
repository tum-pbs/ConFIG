#usr/bin/python3

#version:0.0.8
#last modified:20240512


from . import *
from .style import *
from ..helper.coding import *
import os,math
from typing import Optional,Union


class FormatLinePlotter():
    """
    A class for formatting and plotting line plots.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the FormatLinePlotter class.
        """
        self.colors=None
        #https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/tree/master/Colours
        self.dash_styles=[ls[1] for ls in DASH_STYLE]
        self.markers=["o","^","s","X","D","*","p","v","P","+"]
        self.__draws=[]
        self.__xlabel="$x$"
        self.__ylabel="$y$"
        self.__ncol_legend=None
        self.__fig_save_path="./output_figs/"
        self.__subfigure_index=None
        self.__fig_size=None
        self.__legend_y=None
        self.__xscale="linear"
        self.__yscale="linear"
        self.__grid=False
        self.__show_legend=True
        
    def show_legend(self,show: bool=True):
        """
        Sets whether to show the legend.

        Parameters:
        - show: Whether to show the legend.
        """
        self.__show_legend=show

    def scatter(self,x,y,label:Optional[str]=None,color_style:Union[None,int,str]=None,mark_style:Union[None,int,str]=None,marker_size=10,alpha=1,**kwargs):
        """
        Adds a scatter plot to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - label: The label for the scatter plot.
        - color_style: The style of the color.
        - mark_style: The style of the marker.
        - marker_size: The size of the marker.
        - alpha: The transparency of the marker.
        - **kwargs: Additional keyword arguments, will pass to plt.scatter.

        Returns:
        None
        """
        self.__draws.append(["scatter",(x,y,label,color_style,mark_style,marker_size,alpha,kwargs)])
    
    def scatter_errorbar(self, x, y, y_error: Optional[Sequence]=None, x_error: Optional[Sequence]=None, label: Optional[str]=None, color_style: Optional[Union[int, str]]=None, mark_style: Optional[Union[int, str]]=None, marker_size=10, alpha=1, eline_width=2, cap_size=4, **kwargs):
        """
        Scatter plot with error bars.

        Args:
            x (array-like): The x-coordinates of the data points.
            y (array-like): The y-coordinates of the data points.
            y_error (Optional[Sequence]=None): The error values for the y-coordinates. If provided, error bars will be plotted vertically.
            x_error (Optional[Sequence]=None): The error values for the x-coordinates. If provided, error bars will be plotted horizontally.
            label (Optional[str]=None): The label for the scatter plot.
            color_style (Optional[Union[int, str]]=None): The color style for the scatter plot.
            mark_style (Optional[Union[int, str]]=None): The marker style for the scatter plot.
            marker_size (int): The size of the markers.
            alpha (float): The transparency of the markers.
            eline_width (int): The width of the error bars.
            cap_size (int): The size of the error bar caps.
            **kwargs: Additional keyword arguments to be passed to the scatter plot.

        Returns:
            None
        """
        self.__draws.append(["scatter_errorbar", (x, y, label, color_style, mark_style, x_error, y_error, marker_size, alpha, eline_width, cap_size)])
    
    def scatter_errorbar(self,x,y,y_error:Optional[Sequence]=None,x_error:Optional[Sequence]=None,label:Optional[str]=None,color_style:Union[None,int,str]=None,mark_style:Union[None,int,str]=None,marker_size=10,alpha=1,eline_width=2,cap_size=4,**kwargs):
        """
        Scatter plot with error bars.

        Parameters:
        - x: x-coordinates of the data points.
        - y: y-coordinates of the data points.
        - y_error: Sequence of y-axis error values.
        - x_error: Sequence of x-axis error values.
        - label: Optional label for the scatter plot.
        - color_style: Optional color style for the scatter plot.
        - mark_style: Optional marker style for the scatter plot.
        - marker_size: Size of the markers.
        - alpha: Transparency of the markers.
        - eline_width: Width of the error bars.
        - cap_size: Size of the error bar caps.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        self.__draws.append(["scatter_errorbar",(x,y,label,color_style,mark_style,x_error,y_error,marker_size,alpha,eline_width,cap_size,kwargs)])

    def black_line(self, x, y, label: Optional[str]=None, dash_style: Optional[Union[int, str]]=None, lw=2, **kwargs):
        """
        Adds a black line plot to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - label: The label for the line plot.
        - dash_style: The style of the line.
        - lw: The linewidth of the line.
        - kwargs: Additional keyword arguments.

        Returns:
        None
        """
        self.__draws.append(["black_line", (x, y, label, dash_style, lw, kwargs)])

    def color_line(self, x, y, label: Optional[str]=None, color_style: Optional[Union[int, str]]=None, dash_style: Optional[Union[int, str]]=None, lw=2, **kwargs):
        """
        Adds a color line plot to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - label: The label for the line plot.
        - color_style: The style of the color.
        - dash_style: The style of the line.
        - lw: The linewidth of the line.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        self.__draws.append(["color_line", (x, y, label, color_style, dash_style, lw, kwargs)])

    def scatter_line(self,x,y,label:Optional[str]=None,color_style:Union[None,int,str]=None,dash_style:Union[None,int,str]=None,mark_style:Union[None,int,str]=None,lw=2,marker_size=10,**kwargs):
        """
        Adds a scatter line plot to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - label: The label for the line plot.
        - color_style: The style of the color.
        - dash_style: The style of the line.
        - mark_style: The style of the marker.
        - lw: The linewidth of the line.
        - **kwargs: Additional keyword arguments.

        Returns:
        None        
        """
        self.__draws.append(["scatter_line",(x,y,label,color_style,dash_style,mark_style,lw,marker_size,kwargs)])

    def color_line_errorbar(self,x,y,y_error:Optional[Sequence]=None,x_error:Optional[Sequence]=None,label:Optional[str]=None,color_style:Union[None,int,str]=None,dash_style:Union[None,int,str]=None,lw=2,eline_width=2,cap_size=4,marker=None,marker_size=10,**kwargs):
        """
        Adds a color line plot with error bars to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - y_error: The error in the y-coordinates.
        - x_error: The error in the x-coordinates.
        - label: The label for the line plot.
        - color_style: The style of the color.
        - dash_style: The style of the line.
        - lw: The linewidth of the line.
        - **kwargs: Additional keyword arguments.

        Returns:
        None 
        """
        self.__draws.append(["color_line_errorbar",(x,y,y_error,x_error,label,color_style,dash_style,lw,eline_width,cap_size,marker,marker_size,kwargs)])

    def color_line_errorshadow(self,x,y,y_error:Optional[Sequence]=None,x_error:Optional[Sequence]=None,label:Optional[str]=None,color_style:Union[None,int,str]=None,dash_style:Union[None,int,str]=None,lw=2,alpha=0.2,**kwargs):
        """
        Adds a color line plot with error shadows to the list of draw commands.

        Parameters:
        - x: The x-coordinates of the data points.
        - y: The y-coordinates of the data points.
        - y_error: The error in the y-coordinates.
        - x_error: The error in the x-coordinates.
        - label: The label for the line plot.
        - color_style: The style of the color.
        - dash_style: The style of the line.
        - lw: The linewidth of the line.
        - **kwargs: Additional keyword arguments.

        Returns:
        None 
        """
        self.__draws.append(["color_line_errorshadow",(x,y,y_error,x_error,label,color_style,dash_style,lw,alpha,kwargs)])

    def get_legend_pos(self,num_legend):
        """
        Calculates the position of the legend.

        Parameters:
        - num_legend: The number of legends.

        Returns:
        - ncol: The number of columns in the legend.
        - legend_y: The y position of the legend.
        """
        ncol=self.__ncol_legend
        legend_y=1.02
        if ncol is None:
            if num_legend <=4:
                ncol=num_legend
            if math.ceil(num_legend/4)-math.ceil(num_legend/3)==0:
                if num_legend%3 ==0:
                    ncol= 3
                elif num_legend%4 ==0:
                    ncol= 4
                elif 3-(num_legend%3) < 4-(num_legend%4):
                    ncol= 3
                else:
                    ncol= 4
            else:
                ncol= 4  
        n_row=math.ceil(num_legend/ncol)
        if n_row ==2:
            legend_y=1.1
        else:
            legend_y=1.02
        return ncol, default(self.__legend_y,legend_y)
    
    def _process_mark_style(self,mark_style):
        if mark_style is None:
            return None
        if isinstance(mark_style,int):
            return self.markers[mark_style]
        else:
            return mark_style
        
    def _process_color_style(self,color_style):
        if color_style is None:
            return None
        if isinstance(color_style,int):
            return self.colors[color_style]
        else:
            return color_style
    
    def _process_dash_style(self,dash_style):
        if dash_style is None:
            return None
        if isinstance(dash_style,int):
            return self.dash_styles[dash_style]
        else:
            return dash_style

    def plot(self):
        """
        Plots the line plots based on the draw commands.
        """
        if self.colors is None:
            self.colors=infinite_colors(len(self.__draws))
        if len(self.colors)<len(self.__draws):
            print("Warning: number of datas is larger than the number of colors, which may result in errors. Try to use a larger color list or `infinite_colors()` function.")
        figs=[]
        labels=[]
        plt.figure(figsize=default(self.__fig_size,(8,5)))
        plt.cla()
        num_legend=0
        for i,draws in enumerate(self.__draws):
            name=draws[0]
            if name =="black_line":
                (x,y,label,dash_style,lw,kwargs)=draws[1]      
                fig=plt.plot(x,y,
                             linestyle=default(self._process_dash_style(dash_style),self.dash_styles[i]),
                             linewidth=lw,color='black',**kwargs)
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1
            elif name == "color_line":
                (x,y,label,color_style,dash_style,lw,kwargs)=draws[1] 
                fig=plt.plot(x,y,
                             linewidth=lw,
                             color=default(self._process_color_style(color_style),self.colors[i]),
                             linestyle=default(self._process_dash_style(dash_style),self.dash_styles[i]),
                             **kwargs)
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1
            elif name == "scatter":
                (x,y,label,color_style,mark_style,marker_size,alpha,kwargs)=draws[1] 
                fig=plt.scatter(x,y,
                                marker=default(self._process_mark_style(mark_style),self.markers[i]),
                                c=default(self._process_color_style(color_style),self.colors[i]),
                                s=marker_size*10,alpha=alpha,**kwargs)   
                if label is not None:
                    figs.append(fig)
                    labels.append(label)
                    num_legend+=1
            elif name == "scatter_errorbar":
                handles=[]
                (x,y,label,color_style,mark_style,x_error,y_error,marker_size,alpha,eline_width,cap_size,kwargs)=draws[1] 
                fig=plt.scatter(x,y,
                                marker=default(self._process_mark_style(mark_style),self.markers[i]),
                                c=default(self._process_color_style(color_style),self.colors[i]),
                                s=marker_size*10,alpha=alpha,
                                **kwargs)
                if x_error is not None:
                    s1=plt.errorbar(x=x,y=y,xerr=x_error,fmt="none",
                                    c=default(self._process_color_style(color_style),self.colors[i]),
                                    elinewidth=eline_width,capsize=cap_size)
                    handles.append(s1)
                if y_error is not None:
                    s2=plt.errorbar(x=x,y=y,yerr=y_error,fmt="none",
                                    c=default(self._process_color_style(color_style),self.colors[i]),
                                    elinewidth=eline_width,capsize=cap_size)
                    handles.append(s2)
                if label is not None:
                    if x_error is not None or y_error is not None:
                        handles.append(fig)
                        figs.append(tuple(handles))
                    else:
                        figs.append(fig)
                    labels.append(label)
                    num_legend+=1                  
            elif name == "scatter_line":
                (x,y,label,color_style,dash_style,mark_style,lw,marker_size,kwargs)=draws[1]            
                fig=plt.plot(x,y,
                             marker=default(self._process_mark_style(mark_style),self.markers[i]),
                             c=default(self._process_color_style(color_style),self.colors[i]),
                             linestyle=default(self._process_dash_style(dash_style),self.dash_styles[i]),
                             markersize=marker_size,linewidth=lw,**kwargs)
                if label is not None:
                    figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1   
            elif name == "color_line_errorbar":
                (x,y,y_error,x_error,label,color_style,dash_style,lw,eline_width,cap_size,marker,marker_size,kwargs)=draws[1] 
                fig=plt.errorbar(x=x,y=y,yerr=y_error,xerr=x_error,
                                 c=default(self._process_color_style(color_style),self.colors[i]),
                                 linestyle=default(self._process_dash_style(dash_style),self.dash_styles[i]),
                                 linewidth=lw,elinewidth=eline_width,capsize=cap_size,marker=marker,marker_size=marker_size,
                                 **kwargs)
                if label is not None:
                    figs.append(fig)
                    labels.append(label)
                    num_legend+=1    
            elif name == "color_line_errorshadow":
                (x,y,y_error,x_error,label,color_style,dash_style,lw,alpha,kwargs)=draws[1] 
                handles=[] 
                if y_error is not None:
                    s1=plt.fill_between(x=x,y1=[y[i]-y_error[i] for i in range(len(y))],y2=[y[i]+y_error[i] for i in range(len(y))],
                                        facecolor=default(self._process_color_style(color_style),self.colors[i]),
                                        alpha=alpha)
                    handles.append(s1)
                if x_error is not None:
                    s2=plt.fill_betweenx(y=y,x1=[x[i]-x_error[i] for i in range(len(x))],x2=[x[i]+x_error[i] for i in range(len(x))],
                                         facecolor=default(self._process_color_style(color_style),self.colors[i]),
                                         alpha=alpha)
                    handles.append(s2)
                fig=plt.plot(x,y,linewidth=lw,
                             color=default(self._process_color_style(color_style),self.colors[i]),
                             linestyle=default(self._process_dash_style(dash_style),self.dash_styles[i]),**kwargs)
                if label is not None:
                    if y_error is not None or x_error is not None:
                        handles.append(fig[0])
                        figs.append(tuple(handles))
                    else:
                        figs.append(fig[0])
                    labels.append(label)
                    num_legend+=1     
            else:
                raise Exception("unknow draw type: {}".format(name))
        if num_legend!=0 and self.__show_legend:
            ncol,legend_y=self.get_legend_pos(num_legend)      
            plt.legend(handles=figs, labels=labels,bbox_to_anchor=(0., legend_y, 1., .102), mode="expand", borderaxespad=0.,ncol=ncol)
        if self.__subfigure_index is not None:
            plt.suptitle(self.__subfigure_index,x=0.01,y=0.88,fontproperties="Times New Roman")
        plt.xscale(self.__xscale)
        plt.yscale(self.__yscale)
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.grid(self.__grid)
        plt.minorticks_on()
        return plt.gca()
            
    def xlabel(self,xlabel):
        """
        Sets the label for the x-axis.

        Parameters:
        - xlabel: The label for the x-axis.
        """
        self.__xlabel=xlabel
    
    def ylabel(self,ylabel):
        """
        Sets the label for the y-axis.

        Parameters:
        - ylabel: The label for the y-axis.
        """
        self.__ylabel=ylabel     
    
    def ncol_legend(self,ncol):
        """
        Sets the number of columns in the legend.

        Parameters:
        - ncol: The number of columns in the legend.
        """
        self.__ncol_legend=ncol
        
    def clear_all(self):
        """
        Clears all the settings and draw commands.
        """
        self.__init__()    

    def fig_save_path (self,path):
        """
        Sets the path to save the figure.

        Parameters:
        - path: The path to save the figure.
        """
        self.__fig_save_path=path  

    def subfigure_index(self,index):
        """
        Sets the index of the subfigure.

        Parameters:
        - index: The index of the subfigure.
        """
        self.__subfigure_index=index

    def fig_size(self,size):
        """
        Sets the size of the figure.

        Parameters:
        - size: The size of the figure.
        """
        self.__fig_size=size

    def legend_y(self,y):
        """
        Sets the y position of the legend.

        Parameters:
        - y: The y position of the legend.
        """
        self.__legend_y=y
        
    def xscale(self,scale):
        """
        Sets the scale of the x-axis.

        Parameters:
        - scale: The scale of the x-axis.
        """
        self.__xscale=scale
    
    def yscale(self,scale):
        """
        Sets the scale of the y-axis.

        Parameters:
        - scale: The scale of the y-axis.
        """
        self.__yscale=scale

    def grid(self,grid):
        """
        Sets whether to show grid lines on the plot.

        Parameters:
        - grid: Whether to show grid lines on the plot.
        """
        self.__grid=grid

    def save(self,filename):
        """
        Saves the plot to a file.

        Parameters:
        - filename: The name of the file to save.
        """
        os.makedirs(self.__fig_save_path,exist_ok=True)
        plt.savefig(self.__fig_save_path+filename+".svg",bbox_inches = 'tight')

    def set_colors(self,colors):
        '''
        set colors for the plotter
        '''
        self.colors=colors

line_plotter=FormatLinePlotter()

class DoubleSidePlotter():
    """
    A class for creating a double-sided plot with two y-axes.

    Attributes:
        left_axis (matplotlib.axes.Axes): The left y-axis.
        right_axis (matplotlib.axes.Axes): The right y-axis.

    Methods:
        set_xlabel(label): Sets the x-axis label.
        fast_plot(data_1, data_2, data_x=None, labels=["data_1", "data_2"], log=True, xlabel:Optional[str]=None, color_data1=None, color_data2=None): Plots the data on the double-sided plot.

    """

    def __init__(self) -> None:
        """
        Initializes the DoubleSidePlotter object.

        Creates a figure and two axes for the left and right y-axes.

        """
        fig, ax_left = plt.subplots()
        self.left_axis = ax_left
        self.right_axis = ax_left.twinx()

    def set_xlabel(self, label):
        """
        Sets the x-axis label.

        Args:
            label (str): The label for the x-axis.

        """
        self.left_axis.set_xlabel(label)

    def fast_plot(self, data_1, data_2, xlabel:Optional[str]=None,data_x=None, labels=["data_1", "data_2"], log=True, color_data1=None, color_data2=None):
        """
        Plots the data on the double-sided plot.

        Args:
            data_1 (array-like): The data for the left y-axis.
            data_2 (array-like): The data for the right y-axis.
            data_x (array-like, optional): The data for the x-axis. Defaults to None.
            labels (list, optional): The labels for the left and right y-axes. Defaults to ["data_1", "data_2"].
            log (bool, optional): Whether to use a logarithmic scale for the y-axes. Defaults to True.
            xlabel (str, optional): The label for the x-axis. Defaults to None.
            color_data1 (str, optional): The color for the data_1 line. Defaults to None.
            color_data2 (str, optional): The color for the data_2 line. Defaults to None.

        """
        color_data1 = default(color_data1, LINE_COLOR[0])
        color_data2 = default(color_data2, LINE_COLOR[1])
        if data_x is not None:
            self.left_axis.plot(data_x, data_1, color=color_data1)
            self.right_axis.plot(data_x, data_2, color=color_data2)
        else:
            self.left_axis.plot(data_1, color=color_data1)
            self.right_axis.plot(data_2, color=color_data2)
        self.left_axis.set_ylabel(labels[0], color=color_data1)
        self.right_axis.set_ylabel(labels[1], color=color_data2)
        self.left_axis.tick_params(axis='y', labelcolor=color_data1)
        self.right_axis.tick_params(axis='y', labelcolor=color_data2)
        if log:
            self.left_axis.set_yscale('log')
            self.right_axis.set_yscale('log')
        if xlabel is not None:
            self.left_axis.set_xlabel(xlabel)
        
def plot_double_side(data_1,data_2,xlabel:Optional[str]=None,data_x=None,labels=["data_1","data_2"],log=True,color_data1=None,color_data2=None):
    '''
    Plots the data on a double-sided plot. See DoubleSidePlotter.fast_plot for more details.

    Args:
        data_1 (array-like): The data for the left y-axis.
        data_2 (array-like): The data for the right y-axis.
        data_x (array-like, optional): The data for the x-axis. Defaults to None.
        labels (list, optional): The labels for the left and right y-axes. Defaults to ["data_1", "data_2"].
        log (bool, optional): Whether to use a logarithmic scale for the y-axes. Defaults to True.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        color_data1 (str, optional): The color for the data_1 line. Defaults to None.
        color_data2 (str, optional): The color for the data_2 line. Defaults to None.
    '''
    DoubleSidePlotter().fast_plot(data_1,data_2,data_x=data_x,labels=labels,log=log,xlabel=xlabel,color_data1=color_data1,color_data2=color_data2)

