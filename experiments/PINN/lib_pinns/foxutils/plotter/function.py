#usr/bin/python3

#version:0.0.11
#last modified:20230906

from .field import show_each_channel
from .style import *

CHANNEL_NAME_PVE=[r"$\mathrm{p}$",r"$\mathrm{u}_x$",r"$\mathrm{u}_y$"]

CHANNEL_NAME_VEP=[r"$\mathrm{u}_x$",r"$\mathrm{u}_y$",r"$\mathrm{p}$"]

CHANNEL_NAME_MEAN=[r"$\boldsymbol{\mu}_{\mathrm{p}^*}$",r"$\boldsymbol{\mu}_{\mathrm{u}_x^*}$",r"$\boldsymbol{\mu}_{\mathrm{u}_y^*}$"]

CHANNEL_NAME_STD=[r"$\boldsymbol{\sigma}_{\mathrm{p}^*}$",r"$\boldsymbol{\sigma}_{\mathrm{u}_x^*}$",r"$\boldsymbol{\sigma}_{\mathrm{u}_y^*}$"]

def plot_p_ve(fields,case_names=None,cmap=CMAP_COOLHOT,save_name=None,channel_names=CHANNEL_NAME_PVE,use_sym_colormap=True):
    show_each_channel(fields=fields,case_names=case_names,cmap=cmap,save_name=save_name,channel_names=channel_names,transpose=True,inverse_y=True,tick_format="%.3f",use_sym_colormap=use_sym_colormap,xspace=0.8)
    
def plot_ve_p(fields,case_names=None,cmap=CMAP_COOLHOT,save_name=None,channel_names=CHANNEL_NAME_VEP,use_sym_colormap=True):
    show_each_channel(fields=fields,case_names=case_names,cmap=cmap,save_name=save_name,channel_names=channel_names,transpose=True,inverse_y=True,tick_format="%.3f",use_sym_colormap=use_sym_colormap,xspace=0.8)
    
def plot_p_ve_mean(fields,case_names=None,cmap=CMAP_COOLHOT,save_name=None,channel_names=CHANNEL_NAME_MEAN,use_sym_colormap=True):
    show_each_channel(fields=fields,case_names=case_names,cmap=cmap,save_name=save_name,channel_names=channel_names,transpose=True,inverse_y=True,tick_format="%.3f",use_sym_colormap=use_sym_colormap,xspace=0.8)

def plot_p_ve_std(fields,case_names=None,cmap=CMAP_HOT,save_name=None,channel_names=CHANNEL_NAME_STD,use_sym_colormap=False):
    show_each_channel(fields=fields,case_names=case_names,cmap=cmap,save_name=save_name,channel_names=channel_names,transpose=True,inverse_y=True,tick_format="%.3f",use_sym_colormap=use_sym_colormap,xspace=0.8)
