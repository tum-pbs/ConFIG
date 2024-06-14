#usr/bin/python3

#version:0.0.5
#last modified:20240220

from . import *
from typing import Sequence

COOL=mlp.cm.get_cmap("coolwarm")(np.linspace(0, 0.5, 5))
HOT=mlp.cm.get_cmap("coolwarm")(np.linspace(0.5, 1, 5))
WHITE=[[1,1,1,1]]

CMAP_COOL=colors.LinearSegmentedColormap.from_list("COOL",np.vstack((COOL[0:-1],WHITE)))
CMAP_HOT=colors.LinearSegmentedColormap.from_list("HOT",np.vstack((WHITE,HOT[1:])))
CMAP_COOLHOT=colors.LinearSegmentedColormap.from_list("HOT",np.vstack((COOL[0:-1],WHITE,HOT[1:])))

LINE_COLOR=['#FF1F5B', '#009ADE', '#FFC61E', '#AF58BA', '#F28522', '#00CD6C','#A6761D']
LINE_COLOR_EXTEND=LINE_COLOR+["#B2A4FF","#96CEB4","#3C486B"]
#https://github.com/OrdnanceSurvey/GeoDataViz-Toolkit/tree/master/Colours

DASH_STYLE = [
    ('solid', (0, ())), 
     ('dashed', (0, (5, 5))),
     ('dotted', (0, (1, 1))), 
    ('dashdot', (0, (3, 5, 1, 5))), 
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5,1, 5))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def sym_colormap(d_min,d_max,d_cen=0,cmap="coolwarm",cmapname="sym_map"):
    '''
    Generate a symmetric colormap.

    Args:
        d_min (float): The minimum value of the colormap.
        d_max (float): The maximum value of the colormap.
        d_cen (float, optional): The center value of the colormap. Defaults to 0.
        cmap (str, optional): The colormap to use. Defaults to "coolwarm".
        cmapname (str, optional): The name of the colormap. Defaults to "sym_map".

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The generated colormap.
    '''
    if abs(d_max-d_cen)>abs(d_min-d_cen):
        max_v=1
        low_v=0.5-(d_cen-d_min)/(d_max-d_cen)*0.5
    else:
        low_v=0
        max_v=0.5+(d_max-d_cen)/(d_cen-d_min)*0.5
    if isinstance(cmap,str):
        cmap=mlp.cm.get_cmap(cmap)
    return colors.LinearSegmentedColormap.from_list(cmapname,cmap(np.linspace(low_v, max_v, 100)))

def generate_colormap_from_list(color_list: Sequence,name:str,qualitative:bool=False):
    if qualitative:
        return colors.ListedColormap(color_list,name=name)
    else:
        '''
        rgba_value=to_rgba_array(color_list)
        segmentdata={
            'red':[(i/(len(rgba_value)-1),c[0],c[0]) for i,c in enumerate(rgba_value)],
            'green':[(i/(len(rgba_value)-1),c[1],c[1]) for i,c in enumerate(rgba_value)],
            'blue':[(i/(len(rgba_value)-1),c[2],c[2]) for i,c in enumerate(rgba_value)],
            'alpha':[(i/(len(rgba_value)-1),c[3],c[3]) for i,c in enumerate(rgba_value)]
        }
        return LinearSegmentedColormap(name,segmentdata=segmentdata)        
        '''
        return colors.LinearSegmentedColormap.from_list(name,color_list)

def infinite_colors(n_colors:int,color_list=LINE_COLOR):
    if n_colors <= len(color_list):
        return color_list[:n_colors]
    else:
        return generate_colormap_from_list(color_list,'infinite_colors_temp',qualitative=False)(np.linspace(0,1,n_colors,endpoint=True))


def enable_print_style(font_name="Times New Roman", font_size=30, enable_latex=True):
    #mlp.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    mlp.rcParams['font.sans-serif'] = [font_name, font_name]
    mlp.rcParams['font.size'] = font_size
    if enable_latex:
        mlp.rcParams['text.usetex'] =True

