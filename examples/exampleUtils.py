#Imports
import os, re, sys, json, pathlib, shutil,  random, math, csv
from glob import glob
import argparse
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import imageio #Bose: Imageio is a Python library that provides an easy interface to read and write a wide range of image data, including animated images, volumetric data
import matplotlib.pyplot as plt
import matplotlib.cm as cm #Bose: matplotlib colormaps and functions
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize #Bose: The matplotlib.colors module is used for converting color or numbers arguments to RGBA or RGB.This module is used for mapping numbers to colors or color specification conversion in a 1-D array of colors also known as colormap.And Normalize class is used to normalize data into the interval of [0.0, 1.0].
from skimage.segmentation import mark_boundaries #Bose: Return image with boundaries between labeled regions highlighted
from skimage.transform import rescale #Bose: Rescale operation resizes an image by a given scaling factor. The scaling factor can either be a single floating point value, or multiple values - one along each axis.
colormap = cm.hsv
norm = Normalize()
import plotly.graph_objects as go


import pydicom
import scipy
from scipy import ndimage
from scipy.io import loadmat
from scipy import  signal
from scipy.ndimage import morphology

import ipywidgets as widgets
from ipywidgets import interactive,interact, interact_manual, HBox, Layout,VBox
from IPython.display import display, clear_output

import interpol
from interpol.api import affine_grid

from functools import partial, reduce

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from viu.io import volume
from viu.io.volume import read_volume
from viu.torch.deformation.fields import DVF, set_identity_mapping_cache
from viu.torch.io.deformation import *
from viu.util.body_mask import seg_body
from viu.util.memory import fmt_mem_size
from viu.util.config import json_config
# from viu.torch.visualization.ortho_utils import save_ortho_views, save_single_views #from pamomo.visualization.ortho_utils import save_ortho_views
from viu.torch.measure.voi import measure_voi_list

from pamomo.pca.cmo_pca import CMoPCA
# from pamomo.registration.deformable import reg, force_unload
from pamomo.visualization.cmo_pca_plots import *
from pamomo.metrices.residual_deformation import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ImportFromNII(fname):
    img = nib.load(fname)
    vol, qform = img.get_fdata(), img.header.get_qform()
    vol = np.transpose(vol, (2, 1, 0))   #necessary to obtain the ZYX order

    diag = np.diag(qform)[0:3]
    rcol = qform[0:3,3]
    res = np.abs(diag)
    pos = np.multiply(np.sign(diag), rcol)

    return vol, res, pos

def printDetails(prefix, vol, res, pos):
    print(f'{prefix} vol shape {vol.shape} dtype {vol.dtype} min {np.min(vol)} max {np.max(vol)}')
    print(f'{prefix} res type {type(res)} value {res}')
    print(f'{prefix} pos type {type(pos)} value {pos}')


def printTensor(name,tensor):
    print(f'{name} shape: {tensor.shape} device: {tensor.device} dtype: {tensor.dtype}')

class v1_volumeComparisonViewer3D:

    def __init__(self, 
        listVolumes, listLabels, 
        maxZ0=256, maxZ1=256, maxZ2=256, 
        figsize=(12,8), cmap='gray', 
        displayColorbar=True, useExternalWindowCenter=False, wMin=-500, wMax=500,
        plotHistogram=False, binEdges=[-1000, -200, -50,  0, 50, 200, 1000], xTicks=[-1000, -200,  -50,  50,  200, 1000],useAspectCol=True):
        assert len(listVolumes)==len(listLabels), f'listVolumes and listLabels should have same number of elements'
        for idx ,volume in enumerate(listVolumes):
            assert volume.shape==(maxZ0, maxZ1, maxZ2), f'listVolumes[{idx}] shape mismatch'

        self.listVolumes=listVolumes
        self.listLabels=listLabels
        self.numVolumes=len(listVolumes)
        # assert self.numVolumes > 1, f'We need atleast 2 volumes.'

        self.maxZ0, self.maxZ1, self.maxZ2 = maxZ0, maxZ1, maxZ2
        self.figsize = figsize
        self.cmap = cmap

        self.displayColorbar=displayColorbar
        self.useExternalWindowCenter=useExternalWindowCenter
        self.wMin=wMin
        self.wMax=wMax
        self.plotHistogram=plotHistogram
        self.binEdges=binEdges
        self.xTicks=xTicks
        self.useAspectCol=useAspectCol

        self.list_v = [[np.min(volume), np.max(volume)] for  volume in self.listVolumes]
        #Mapping from displayPlane Name to [displayPlane_index, minSliceIndex_inPlane, maxSliceIndex_inPlane, defaultSliceIndex_inPlane]
        self.planeInfoDict = {
            '0_PlaneHorDim3VerDim2':[0, 0, maxZ0-1,  maxZ0//2],
            '1_PlaneHorDim3VerDim1':[1, 0, maxZ1-1,  maxZ1//2],
            '2_PlaneHorDim2VerDim1':[2, 0, maxZ2-1,  maxZ2//2]
        }
        self.infoList_0 = self.planeInfoDict['0_PlaneHorDim3VerDim2']
        self.infoList_1 = self.planeInfoDict['1_PlaneHorDim3VerDim1']
        self.infoList_2 = self.planeInfoDict['2_PlaneHorDim2VerDim1']
        # Call to select slice plane
        widgets.interact(self.views)

    def views(self):
        # widgets.Text(value=self.addlText,placeholder='Type something',description='String:',disabled=False )
        # widgets.interact(self.plot_slice,
        #     z0=widgets.IntSlider(min=self.infoList_0[1], max=self.infoList_0[2], value=self.infoList_0[3], step=1, continuous_update=False, description=f'0_PlaneHorDim3VerDim2'),
        #     z1=widgets.IntSlider(min=self.infoList_1[1], max=self.infoList_1[2], value=self.infoList_1[3], step=1, continuous_update=False, description=f'1_PlaneHorDim3VerDim1'),
        #     z2=widgets.IntSlider(min=self.infoList_2[1], max=self.infoList_2[2], value=self.infoList_2[3], step=1, continuous_update=False, description=f'2_PlaneHorDim2VerDim1'),
        #     )
        from ipywidgets import IntSlider, SliderStyle
        redStyle = SliderStyle(handle_color='red')
        yellowStyle = SliderStyle(handle_color='yellow')
        greenStyle = SliderStyle(handle_color='green')

        # widget=interactive(self.plot_slice, z0=128, z1=128, z2=128)
        widget=interactive(self.plot_slice,
            z0=widgets.IntSlider(min=self.infoList_0[1], max=self.infoList_0[2], value=self.infoList_0[3], step=1, continuous_update=False, style=redStyle, description=f'0_PlaneHorDim3VerDim2'),
            z1=widgets.IntSlider(min=self.infoList_1[1], max=self.infoList_1[2], value=self.infoList_1[3], step=1, continuous_update=False, style=yellowStyle, description=f'1_PlaneHorDim3VerDim1'),
            z2=widgets.IntSlider(min=self.infoList_2[1], max=self.infoList_2[2], value=self.infoList_2[3], step=1, continuous_update=False, style=greenStyle, description=f'2_PlaneHorDim2VerDim1'),
            )
        controls = HBox(widget.children[:-1], layout = Layout(flex_flow='row wrap'))
        output = widget.children[-1]
        display(VBox([controls, output]))

    def computeAndPlotSliceHistogram(self, sliceImage, binEdges, xTicks, ax_ij):
        #https://realpython.com/python-histograms/
        #https://stackoverflow.com/questions/35418552/aligning-bins-to-xticks-in-plt-hist
        d= sliceImage.flatten()
        n, bins, patches = ax_ij.hist(x=d, bins=binEdges, color='#0504aa',  alpha=0.7, rwidth=0.85)
        ax_ij.grid(axis='y', alpha=0.75)
        ax_ij.set_xticks(xTicks)
        ax_ij.set_xlabel('HU Diff')
        ax_ij.set_ylabel('Num Voxels')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        ax_ij.set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    def plot_slice(self, z0, z1, z2):
        # Plot slice for the given plane and slice
        numRows=self.numVolumes if False== self.plotHistogram else 2*self.numVolumes #3
        numCols=3 #self.numVolumes
        f,ax = plt.subplots(numRows,numCols, squeeze=False, figsize=self.figsize)
        for col in range(3):
            if 0==col:
                listOfDiaplaySlicesInCol = [volume[z0,:,:] for volume in self.listVolumes]
            if 1==col:
                #But coronal and sagittal views are upside down when displayed original. So compensate.
                listOfDiaplaySlicesInCol = [np.flipud(volume[:,z1,:]) for volume in self.listVolumes]
            if 2==col:
                #But coronal and sagittal views are upside down when displayed original. So compensate.
                listOfDiaplaySlicesInCol = [np.flipud(volume[:,:,z2]) for volume in self.listVolumes]
            aspect_col = f'{(listOfDiaplaySlicesInCol[0].shape[1]+0.01) / (listOfDiaplaySlicesInCol[0].shape[0]+0.01):.1f}'
            # print(f'col: {col} aspect_col: {aspect_col}')
            for row in range(self.numVolumes):
                img_row = row if False== self.plotHistogram else 2*row
                im_ax_ij = ax[img_row,col]
                im_ax_ij.grid(False)
                # aspect_row_col = f'{(listOfDiaplaySlicesInCol[row].shape[1]+0.01) / (listOfDiaplaySlicesInCol[row].shape[0]+0.01):.1f}'
                # https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
                if self.useExternalWindowCenter:
                    vminVal, vmaxVal = self.wMin, self.wMax
                else:
                    vminVal, vmaxVal = self.list_v[row][0], self.list_v[row][1]
                # im=im_ax_ij.imshow(listOfDiaplaySlicesInCol[row], cmap=plt.get_cmap(self.cmap), vmin=self.list_v[row][0], vmax=self.list_v[row][1], interpolation='none', aspect=aspect_col) # #, aspect=aspect_row_col
                if True==self.useAspectCol:
                    im=im_ax_ij.imshow(listOfDiaplaySlicesInCol[row], cmap=plt.get_cmap(self.cmap), vmin=vminVal, vmax=vmaxVal, interpolation='none', aspect=aspect_col) # #, aspect=aspect_row_col
                else:
                    im=im_ax_ij.imshow(listOfDiaplaySlicesInCol[row], cmap=plt.get_cmap(self.cmap), vmin=vminVal, vmax=vmaxVal, interpolation='none') # #, aspect=aspect_row_col

                if self.displayColorbar:
                    plt.colorbar(im, ticks=self.xTicks, ax=im_ax_ij)
                if 0==col:
                    im_ax_ij.set_title(self.listLabels[row],fontsize=10)
                    im_ax_ij.axhline(y=z1, xmin=self.infoList_1[1], xmax=self.infoList_1[2],color='yellow',ls=':',lw=1.5)
                    im_ax_ij.axvline(x=z2, ymin=self.infoList_2[1], ymax=self.infoList_2[2],color='green',ls=':',lw=1.5)
                if 1==col:
                    # Because in coronal view increasing column  moves towards feet but  increasing z0 in axial view takes towards head
                    # Representation of  z0, i.e. red line location  in coronal view should be given at imageHeight -z0
                    # = (maxZ0-1) - z0 = (self.infoList_0[2] - 1)-z0
                    im_ax_ij.axhline(y=((self.infoList_0[2] - 1)-1)-z0, xmin=self.infoList_0[1], xmax=self.infoList_0[2],color='red',ls=':',lw=1.5)
                    im_ax_ij.axvline(x=z2, ymin=self.infoList_2[1], ymax=self.infoList_2[2],color='green',ls=':',lw=1.5)
                if 2==col:
                    # Because in coronal view increasing column  moves towards feet but  increasing z0 in axial view takes towards head
                    # Representation of  z0, i.e. red line location  in coronal view should be given at imageHeight -z0
                    # = (maxZ0-1) - z0 = (self.infoList_0[2] - 1)-z0
                    im_ax_ij.axhline(y=(self.infoList_0[2] - 1)-z0, xmin=self.infoList_0[1], xmax=self.infoList_0[2],color='red', ls=':',lw=1.5)
                    im_ax_ij.axvline(x=z1, ymin=self.infoList_1[1], ymax=self.infoList_1[2],color='yellow',ls=':',lw=1.5)
                if True==self.plotHistogram:
                    histogramRow = img_row+1
                    sliceImage=listOfDiaplaySlicesInCol[row]
                    hist_ax_ij = ax[histogramRow,col]
                    self.computeAndPlotSliceHistogram(sliceImage, self.binEdges, self.xTicks, hist_ax_ij)
        # plt.axis("scaled")
        plt.show()

def getPushRotationMatrix(theta_deg, viewString,
        center_slice_z, center_row_y, center_col_x,
        tra_z, tra_y, tra_x):
    """
    theta_deg: SINGLE angle of rotation along ONE viewString axis
    viewString : axial, coronal, sagittal   (along Z, Y, X axis respectively)

    Center of rotation is given by center_slice_z, center_row_y, center_col_x.
    center_slice_z : center along 1st / Z dimension
    center_row_y : center along 2nd / Y dimensioon
    center_col_x : center along 3rd / X dimension

    Order of operation is first rotation, then translation where the translation
    is along the rotated axis. So the order of operation is
    1. Translate to origin
    2. Rotate
    3. Translate back to center
    4. Translate along the rotated axis
    tra_z, tra_y, tra_x : translation along Z, Y, X axis respectively

    For  understanding  clockwise or counter-clockwise rotation direction
    in the view direction, read the description below.

    Rotation around Dicom Z axis : As viewed into Axial plane    from origin
    toward +Z axis (towards H(Head)), with A (Anterior) on top of the view,
    TAKE clockwise rotation as positive

    Rotation around Dicom Y axis : As viewed into Coronal plane  from origin
    toward +Y axis (towards P(Posterior)), with H (Head) on top of the view,
    TAKE  clockwise rotation as positive

    Rotation around Dicom X axis : As viewed  (lying down horizontally) into
    Sagittal plane from +X toward origin (Rowards R(Right)) with H (Head) on
    top of the view, TAKE COUNTER-CLOCKWISE rotation as  positive AS  THE 
    VIEW IS LOOKING TOWARDS ORIGIN FROM +X Axis

    Internally  use the standard RH Coordinate system rotationmatrix as described in
    wikipedia: https://en.wikipedia.org/wiki/Rotation_matrix

    """
    assert viewString in ["axial", "coronal", "sagittal"],\
        f'viewString {viewString} not in ["axial", "coronal", "sagittal"]'
    # rad = np.radians(theta_deg) if viewString in ["axial", "coronal"]\
    #     else 0 - np.radians(theta_deg)
    rad = np.radians(theta_deg)

    if "axial"==viewString:
        rot = np.array([
            [1.,            0,            0,  0],
            [ 0,  np.cos(rad), -np.sin(rad),  0],
            [ 0,  np.sin(rad),  np.cos(rad),  0],
            [ 0,            0,            0, 1.] ],'float32')
    if "coronal"==viewString:
        rot = np.array([
            [  np.cos(rad),  0,  np.sin(rad),  0],
            [            0, 1.,            0,  0],
            [ -np.sin(rad),  0,  np.cos(rad),  0],
            [            0,  0,            0, 1.] ], 'float32')
    if "sagittal"==viewString:
        rot = np.array([
            [ np.cos(rad), -np.sin(rad),  0,  0],
            [ np.sin(rad),  np.cos(rad),  0,  0],
            [           0,            0, 1.,  0],
            [           0,            0,  0, 1.] ], 'float32')
    # print(f'rot: {rot}')
    tra_to_center = np.array([
        [ 1, 0, 0, -center_slice_z],
        [ 0, 1, 0,   -center_row_y],
        [ 0, 0, 1,   -center_col_x],
        [ 0, 0, 0,              1.]], 'float32')
    # print(f'transToOrigin: {tra_to_center}')
    tra_to_origin = np.array([
        [ 1, 0, 0, center_slice_z],
        [ 0, 1, 0,   center_row_y],
        [ 0, 0, 1,   center_col_x],
        [ 0, 0, 0,             1.]], 'float32')
    # print(f'transToCenter: {tra_to_origin}')
    tra_final = np.array([
        [ 1, 0, 0, tra_z],
        [ 0, 1, 0, tra_y],
        [ 0, 0, 1, tra_x],
        [ 0, 0, 0,  1.]], 'float32')
    # print(f'transFinal: {tra_final}')

    #The final transformation is described as A(x-c)+t+c
    #A(x-c) = rot @ tra_to_center @ x
    #tra_final @ tra_to_origin @ rot @ tra_to_center @ x = A(x-c)+t+c

    #Note however our rot-matrix formulation assumes X-Y-Z co-ordinate sequence 
    # but the volume / points we pass have a Z-Y-X layout. So we use 
    # A = np.transpose(rot) instead of rot.
    pushAffine_np= tra_final @ tra_to_origin @ np.transpose(rot) @ tra_to_center
    return pushAffine_np

def getPyTorchAffineMatTensor(unNormalizedAffineMatImageCoord_a2b_np,
        depth_a, height_a, width_a,
        depth_b, height_b, width_b,
        device=torch.device('cpu')):
    """
    parameters:
    unNormalizedAffineMatImageCoord_a2b_np : 4x4 (homogeneous)
    affine matrix, used in torch-interpol,  to be multiplied with
    pixel location (row, col) = (i,j) of location
    in volume a to obtain location in volume b,
    depth_a, height_a, width_a: depth, height, width  of vol a
    depth_b, height_b, width_b:  depth, height, width  of vol b
    returnPyTorchTheta: If true, it also returns pyTorch Theta
    (to be used by F.affine_grid or VIU DVF.affine).

    It should be noted that pyTorch Theta is a tensor version of
    normalizedAffineMat but to be applied on normalised (x,y)
    instead of un-normalised pixel co-ordinate (i,j) as is the norm in
    torch-interpol. Therefore, after normalization the rows of the matrix are
    swapped.

    Finally the last homogeneous row is not required in

    return:
    1. normalized affine matrix  that is to be multiplied with normalized
    pixel location (y,x) (from -1 to +1) of image a to obtain
    normalized pixel location in image b.
    """
    T_normalized2Regular_a_yx = np.linalg.inv(np.array(
        [[2./depth_a,           0,          0, -1.],
         [         0, 2./height_a,          0, -1.],
         [         0,           0, 2./width_a, -1.],
         [         0,           0,          0,  1.]], 'float32'))
    T_regular2Normalized_b_yx = np.array(
        [[2./depth_b,           0,          0, -1.],
         [         0, 2./height_b,          0, -1.],
         [         0,           0, 2./width_b, -1.],
         [         0,           0,          0,  1.]], 'float32')

    #Normalize

    pyTorchAffineMatTensor =\
          torch.from_numpy(T_regular2Normalized_b_yx @\
            unNormalizedAffineMatImageCoord_a2b_np @\
            T_normalized2Regular_a_yx).to(device)[0:3,:]
    # In rows and columens: 0->2, 2->0, 1->1
    # [0,0], [0,1], [0,2],
    # [1,0], [1,1], [1,2],
    # [2,0], [2,1],[2,2]
    block3x3Flipped  =  pyTorchAffineMatTensor[0:3, 0:3].flip([0,1])
    lastColFlipped =  pyTorchAffineMatTensor[0:3,-1].flip(0)
    pyTorchAffineMatTensor[0:3, 0:3]=block3x3Flipped
    pyTorchAffineMatTensor[0:3,-1]=lastColFlipped
    return pyTorchAffineMatTensor


def getUnNormalizedAffineMatTensorInImageCoord(pyTorchAffineMatTensorInNormalizedCoord_a2b,
        depth_a, height_a, width_a,
        depth_b, height_b, width_b):
    """
    parameters:
    pyTorchAffineMatTensorInNormalizedCoord_a2b : 4x4 (homogeneous) affine matrix in normalized image coordinate
    depth_a, height_a, width_a: depth, height, width  of vol a
    depth_b, height_b, width_b:  depth, height, width  of vol b

    return:
    1. un-normalized affine matrix  that is to be used by torch-interpol to create affine grid

    """
    device = pyTorchAffineMatTensorInNormalizedCoord_a2b.device
    block3x3Flipped = pyTorchAffineMatTensorInNormalizedCoord_a2b[0:3, 0:3].flip([0,1])
    lastColFlipped =  pyTorchAffineMatTensorInNormalizedCoord_a2b[0:3,-1].flip(0)
    tmpAffineMat=pyTorchAffineMatTensorInNormalizedCoord_a2b.clone()
    tmpAffineMat[0:3, 0:3]=block3x3Flipped
    tmpAffineMat[0:3,-1]=lastColFlipped

    T_regular2normalized_a_yx = torch.tensor(
        [[2./depth_a,           0,          0, -1.],
         [         0, 2./height_a,          0, -1.],
         [         0,           0, 2./width_a, -1.],
         [         0,           0,          0,  1.]],
         dtype=torch.float32).to(device)
    T_normalized2regular_b_yx = torch.tensor(
        [[2./depth_b,           0,          0, -1.],
         [         0, 2./height_b,          0, -1.],
         [         0,           0, 2./width_b, -1.],
         [         0,           0,          0,  1.]],
         dtype=torch.float32).to(device).inverse()
    unNormalizedAffineMatTensorInImageCoord =\
        T_normalized2regular_b_yx.matmul(tmpAffineMat.matmul(T_regular2normalized_a_yx))
    return unNormalizedAffineMatTensorInImageCoord

#Generate normalized torch-functional  grid   from  torch-interpol grid.
def convertGrid_interpol2functional(interpol_grid_batched, depth, height, width):
    # printTensor("interpol_grid_batched", interpol_grid_batched)
    batchSize=interpol_grid_batched.shape[0]
    #normalization matrix
    normalizationMat = torch.tensor(
        [[2./depth,         0,        0, -1.],
         [       0, 2./height,        0, -1.],
         [       0,         0, 2./width, -1.],
         [       0,         0,        0,  1.]],
        dtype=torch.float32, device=interpol_grid_batched.device)
    nb_dim = normalizationMat.shape[-1] - 1
    normalizationMat_rot = normalizationMat[:nb_dim, :nb_dim]
    normalizationMat_tr = normalizationMat[:nb_dim, -1]
    #Expand normalization matrix by batchSize
    normalizationMat_rot = normalizationMat_rot.expand(batchSize, *normalizationMat_rot.shape)
    # printTensor("normalizationMat_rot", normalizationMat_rot)
    normalizationMat_tr =   normalizationMat_tr.expand(batchSize, *normalizationMat_tr.shape)
    # printTensor("normalizationMat_tr", normalizationMat_tr)
    # Add dimension (in-place) in the end to support  matmul with normalizationMat_rot.
    # Then remove that dimension before adding  with normalizationMat_tr
    field_ij_batched_normalized =\
        torch.matmul(normalizationMat_rot, interpol_grid_batched.unsqueeze(-1)).squeeze(-1)\
        + normalizationMat_tr
    # ij to xy
    field_xy_batched_normalized = torch.flip(field_ij_batched_normalized, [-1])
    return field_xy_batched_normalized

#Generate normalized torch-functional  grid   from  torch-interpol grid.
def convertGrid_functional2interpol(functional_grid_batched, depth, height, width):
    # printTensor("functional_grid_batched", functional_grid_batched)
    batchSize=functional_grid_batched.shape[0]
    #xy to ij
    field_ij_batched_normalized = torch.flip(functional_grid_batched, [-1])
    #deNormalization matrix
    deNormalizationMat = torch.linalg.inv(torch.tensor(
        [[2./depth,         0,        0, -1.],
         [       0, 2./height,        0, -1.],
         [       0,         0, 2./width, -1.],
         [       0,         0,        0,  1.]],
        dtype=torch.float32, device=functional_grid_batched.device))
    nb_dim = deNormalizationMat.shape[-1] - 1
    deNormalizationMat_rot = deNormalizationMat[:nb_dim, :nb_dim]
    deNormalizationMat_tr = deNormalizationMat[:nb_dim, -1]
    #Expand deNormalization matrix by batchSize
    deNormalizationMat_rot = deNormalizationMat_rot.expand(batchSize, *deNormalizationMat_rot.shape)
    deNormalizationMat_tr =   deNormalizationMat_tr.expand(batchSize, *deNormalizationMat_tr.shape)
    # Add dimension (in-place) in the end to support  matmul with normalizationMat_rot.
    # Then remove that dimension before adding  with normalizationMat_tr
    field_ij_batched_deormalized =\
        torch.matmul(deNormalizationMat_rot, field_ij_batched_normalized.unsqueeze(-1)).squeeze(-1)\
        + deNormalizationMat_tr
    return field_ij_batched_deormalized

def seg_body_torch(vol, air_threshold=-300):
    try:
        from cc_torch import connected_components_labeling
        shape = (vol < air_threshold).to(torch.uint8).cuda().squeeze().shape

        s = torch.tensor(shape[::-1])[..., None]
        p = tuple(torch.hstack((torch.zeros_like(s), torch.fmod(s, 2))).flatten().tolist())
        m = torch.nn.functional.pad((vol < air_threshold).to(torch.uint8).cuda().squeeze(), p)
        cc = connected_components_labeling(m)
        idx = cc.unique()
        idx = idx[idx != 0][0]  # select index of largest segment that is not zero
        cc = connected_components_labeling((cc != idx).to(torch.uint8))
        idx = cc.unique()
        idx = idx[idx != 0][0]  # select index of largest segment that is not zero
        cc = cc == idx

        for i in range(cc.ndim):
            cc = cc.index_select(i, torch.arange(shape[i], dtype=torch.int64, device=cc.device))

        return morphology.binary_dilation(cc.cpu(), iterations=2)
    except:
        return seg_body(vol.numpy(), air_threshold=air_threshold).astype(np.int8)

def sort_by_series_number(dir_list):
    pattern = re.compile('WFBPOpt(.{2})PercentLongCycles(.*)')
    fnd = {}
    for dn in dir_list:
        fl = glob(os.path.join(dn, '*.dcm'))
        if len(fl) > 0:
            with pydicom.dcmread(fl[0]) as dcm:
                idx = None
                if 'ReconstructionMethod' in dcm:
                    m = pattern.match(dcm.ReconstructionMethod)
                    if m is not None:
                        method, state = m.groups()
                        if method == 'PB':
                            idx = int(state)
                        elif method == 'AB':
                            if state.startswith('Inh'):
                                idx = int(state[3:])
                            elif state.startswith('Exh'):
                                idx = 200 - int(state[3:])
                if idx is not None:
                    fnd[idx] = dn
                else:
                    fnd[dcm.SeriesNumber] = dn
    return [fnd[sn] for sn in sorted(fnd.keys())]

def generateGifFile(patientParentFolder, patientMRN, behaviourPrefixedConfigKey, vols, diff_vols, res, pos, fps, logFilepath):
    # #OLD: If save_single_views NOT implemented
    # # from viu.torch.visualization.ortho_utils import save_ortho_views #from pamomo.visualization.ortho_utils import save_ortho_views
    # #NEW: If save_single_views implemented
    # from viu.torch.visualization.ortho_utils import save_ortho_views, save_single_views 
    #NOTE NEWEST
    from viu.torch.visualization.ortho_utils import save_ortho_views
    ######## LOG #####
    logString = f'Creating gif files for {behaviourPrefixedConfigKey}_{patientMRN}'
    print(logString)
    with open(logFilepath, 'r+') as f:
        f.seek(0)
        f.writelines([logString])
        f.truncate()
        f.close()
    ###################
    gifInputOutputFolder = os.path.join(patientParentFolder,f'gifFolder')# gifInputOutputFolder = os.path.join(patientParentFolder,f'{patientMRN}/gifFolder/')
    os.makedirs(gifInputOutputFolder, exist_ok=True)
    print(f'gifInputOutputFolder {gifInputOutputFolder}')

    print(f'vols type {type(vols)} shape {vols.shape} dtype {vols.dtype}')
    numPhases = vols.shape[0]
    print(f'res type {type(vols)} value {res} dtype {res.dtype}')
    print(f'pos type {type(pos)} value {pos} dtype {pos.dtype}')
    cfg_jsonPath = os.path.join(patientParentFolder,f'{patientMRN}/{behaviourPrefixedConfigKey}_{patientMRN}_reconViews.json')
    print(f'jsonPath: {cfg_jsonPath}')
    ######
    cfg = json_config(cfg_jsonPath)  #NOTE create json file for edge measurement
    if 'views' not in cfg.keys:
        cfg.add_config_item('views', [{'ctr': pos.tolist(), 'voi': None, 'wl': [500, 0]}])
        cfg.write()

    if 'edge_measurements' not in cfg.keys:
        cfg.add_config_item('edge_measurements', [])
        cfg.write()
    #Create frames
    listOfFrameFilePaths=[]
    xView_listOfFrameFilePaths=[]
    diff_listOfFrameFilePaths=[] #<<<<<<<< NOTE
    for phaseIdx in range(numPhases):
        # ############### OLD: ############
        # # pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view.png'
        # # xView_pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_xView.png'
        # # pngFilePath = os.path.join(gifInputOutputFolder,pngFileName)
        # # xView_pngFilePath = os.path.join(gifInputOutputFolder,xView_pngFileName)
        # # save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', vols[phaseIdx,...], res, pos,
        # #                         dst_path=gifInputOutputFolder, fn=pngFileName, views=cfg.views, additionalSingleViewToSave='x',additional_fn=xView_pngFileName)
        # ############### NEW: ############
        # pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view.png'
        # # xView_pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_xView.png'
        # # xView_pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_F_singleView.png'
        # xView_pngFileName = f'F_{pngFileName}'
        # pngFilePath = os.path.join(gifInputOutputFolder,pngFileName)
        # xView_pngFilePath = os.path.join(gifInputOutputFolder,xView_pngFileName)
        # # save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', vols[phaseIdx,...], res, pos,
        # #                         dst_path=gifInputOutputFolder, fn=pngFileName, views=cfg.views, additionalSingleViewToSave='x',additional_fn=xView_pngFileName)
        # save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', vols[phaseIdx,...], res, pos,
        #                         dst_path=gifInputOutputFolder, fn=pngFileName, views=cfg.views)
        # # fnCommon='singleView.png'
        # fnCommon=pngFileName
        # save_single_views( f'{phaseIdx}:{behaviourPrefixedConfigKey}', vols[phaseIdx,...], res, pos,
        #     ctr=cfg.views[0]['ctr'], voi=cfg.views[0]['voi'], centered=False, wl=cfg.views[0]['wl'], diff=False,
        #     fnCommon=fnCommon, dst_path=gifInputOutputFolder, ovl=None, ovl_alpha=0.2, ovl_fig=None, ovl_fig_quad=3,
        #     draw_bounding_boxes_flag=False,
        #     list_boundingBoxDict = [],
        #     single_views_to_save_string='F')
        #################################
        #NOTE NEWEST
        title=f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view'
        pngFileName=f'{title}.png'
        xView_pngFileName=f'F_{title}.png'
        pngFilePath = os.path.join(gifInputOutputFolder,pngFileName)
        xView_pngFilePath = os.path.join(gifInputOutputFolder,xView_pngFileName)
        save_ortho_views(title, vols[phaseIdx,...], res, pos, dst_path=gifInputOutputFolder, views=cfg.views, save_planes='CF')
        #################################
        listOfFrameFilePaths.append(pngFilePath)
        xView_listOfFrameFilePaths.append(xView_pngFilePath)

        #################################
        # diff_pngFileName = f'diff_{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view.png'
        # diff_pngFilePath = os.path.join(gifInputOutputFolder,diff_pngFileName)
        # save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', diff_vols[phaseIdx,...], res, pos,
        #                         dst_path=gifInputOutputFolder, fn=diff_pngFileName, views=cfg.views)
        #################################
        #NOTE NEWEST
        diff_title= f'diff_{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view'
        diff_pngFileName=f'{diff_title}.png'
        diff_pngFilePath= os.path.join(gifInputOutputFolder,diff_pngFileName)
        save_ortho_views(diff_title, diff_vols[phaseIdx,...], res, pos, dst_path=gifInputOutputFolder,views=cfg.views)
        #################################
        diff_listOfFrameFilePaths.append(diff_pngFilePath)

    del vols
    del diff_vols
    #Create gif from frames
    images = []
    for filePath in listOfFrameFilePaths:
        images.append(imageio.imread(filePath))
        os.remove(filePath)
    gifFileName = f'{patientMRN}_fps_{fps:02d}_{behaviourPrefixedConfigKey}.gif'
    gifFilePath = os.path.join(gifInputOutputFolder,gifFileName)
    imageio.mimsave(gifFilePath, images, fps=fps, loop=0)
    print(f'Created {gifFilePath}')

    diff_images = []
    for filePath in diff_listOfFrameFilePaths:
        diff_images.append(imageio.imread(filePath))
        os.remove(filePath)
    diff_gifFileName = f'diff_{patientMRN}_fps_{fps:02d}_{behaviourPrefixedConfigKey}.gif'
    diff_gifFilePath = os.path.join(gifInputOutputFolder,diff_gifFileName)
    imageio.mimsave(diff_gifFilePath, diff_images, fps=fps, loop=0)
    print(f'Created {diff_gifFilePath}')

    xView_images = []
    for filePath in xView_listOfFrameFilePaths:
        xView_images.append(imageio.imread(filePath))
        os.remove(filePath)
    xView_gifFileName = f'{patientMRN}_fps_{fps:02d}_{behaviourPrefixedConfigKey}_xView.gif'
    xView_gifFilePath = os.path.join(gifInputOutputFolder,xView_gifFileName)
    imageio.mimsave(xView_gifFilePath, xView_images, fps=fps, loop=0)
    print(f'Created {xView_gifFilePath}')

def saveOriginalAndCoronalTiff(vols, hdfFilePath):
    # https://forum.image.sc/t/saving-an-nd-image-with-pyimagej/52497/2
    # tifffile directly or via scikit-image works great for such cases. Here you first have to re-arrange your 
    # dimensions to ZCYX order and then use tifffile.imwrite:
    import tifffile
    tiffFilePath = hdfFilePath.replace('.hdf', '.tiff')
    tifffile.imwrite(tiffFilePath, np.moveaxis(vols, 0,1), imagej=True, metadata={'axes': 'ZCYX'})
    tiffFilePath_coronal = hdfFilePath.replace('.hdf', '_coronal.tiff')
    tifffile.imwrite(tiffFilePath_coronal, np.flip(vols.transpose((3,0,1,2)),axis=2), imagej=True, metadata={'axes': 'ZCYX'})

#Move to exampleUtils
def read_pca_write_ycomp(pca_fn, vol_idx_msk, prefilter):
    import tifffile
    pca = CMoPCA(example_msk=vol_idx_msk,prefilter=prefilter)
    pca_mean, pca_res, pca_pos = pca.read(pca_fn, device='cuda')
    print(f'pca_mean type {pca_mean.type} shape {pca_mean.shape} dtype {pca_mean.dtype} device {pca_mean.device}')
    print(f'pca_res shape {pca_res.shape} dtype {pca_res.dtype} ')
    print(f'pca_pos shape {pca_pos.shape} dtype {pca_pos.dtype} ')
    pca_u=pca.u
    print(f'pca_u type {pca_u.type} shape {pca_u.shape} dtype {pca_u.dtype} device {pca_u.device}')
    pca_s=pca.s
    print(f'pca_s type {pca_s.type} shape {pca_s.shape} dtype {pca_s.dtype} device {pca_s.device}')
    pca_vt=pca.vt
    print(f'pca_vt type {pca_vt.type} shape {pca_vt.shape} dtype {pca_vt.dtype} device {pca_vt.device}')
    pca_all_comp_channel_y = pca_u[:,:,:,:,1].cpu().numpy()
    print(f'pca_all_comp_channel_y shape {pca_all_comp_channel_y.shape} dtype {pca_all_comp_channel_y.dtype} ')
    pca_all_comp_channel_y_coronal_tiffFilePath = pca_fn.replace('.hdf', '_pca_all_comp_channel_y_coronal.tiff')
    tifffile.imwrite(pca_all_comp_channel_y_coronal_tiffFilePath, np.flip(pca_all_comp_channel_y.transpose((3,0,1,2)),axis=2), imagej=True, metadata={'axes': 'ZCYX'})

#Example from Igor's code
def do_simple_fp(vol, vol_res, vol_pos, angles, geo, pname, hu_offset=-1000, hu_slope=50000):
    import torch
    from viu.io import volume
    from viu.io import hdfproj
    from viu.torch.varian_reco_ext.varian_reco import VForwardprojectionModule
    #from viu.util.cbct_geometry import CBCTGeometry
    #variable geo is of type CBCTGeometry
    prj_cnt = len(angles)
    print(f'FP of input volume: {np.shape(vol)} {vol_res} {vol_pos}')
    vol = volume.convert_unit(vol, unit = 'HU', to_unit='Attenuation', offset=hu_offset, slope = hu_slope)
    proj_stack_shape = [prj_cnt, geo.prjSize[1], geo.prjSize[0]]
    device = torch.device("cuda:0")
    vol_shape = vol.shape[::-1]
    fpm = VForwardprojectionModule(proj_stack_shape, geo.prjRes, vol_shape, vol_res, vol_tr=vol_pos,
                                   sad=geo.sad, use_half_fan_weighting=geo.half_fan).to(device, torch.float32)
    iso2pix = torch.tensor(geo.createIso2Pix(angles)[np.newaxis, ...]).cuda()
    #print(iso2pix)
    fpm.set_iso2pix(iso2pix, prj_cnt)
    vol = torch.tensor(vol[None, ...]).cuda()
    prjs = fpm(vol)
    projs = prjs.detach().squeeze().cpu().numpy()
    hdfproj.write_proj(pname, projs, iso2pix.detach().squeeze().cpu().numpy(), geo.prjRes,
                        angles, geo.sad, geo.sid, geo.imagerLat, geo.half_fan, geo.scanVelocity, overwrite=True)
    return projs

# #From Sangyu's message in teams
# # Get the projection transform matrix
# with h5py.File(projection_file_path) as F:

# 		P = F["Projection"].attrs["IsoToPixelTransform"].transpose()
# # Get the point position relative to isocenter in world coordinate
# PP = np.array([X, Y, Z, 2]) - np.array([X_iso , Y_iso, Z_iso, 1]) 
# # Calculate
# PPP = P @ PP
# # 2D pixel position
# PPP[:2]/PPP[3]

#Create a function for computing forward projections
def generateForwardProjections(
        vol_data_ZYX,
        vol_res_XYZ_mm,
        vol_pos_XYZ_mm, #this is center of CT volume in patient co-ordinate system
        iso_pos_XYZ_mm, #Ideally this is  planned isocenter in  CT volume in patient co-ordinate system, the point that aligns with machine isocenter
        alreadyNormalizedFlag=False,
        hu_offset=-1000,
        hu_max=400,
        hu_slope=50000,
        proj_angles_degrees=np.array([0.0, 90.0], 'float32'),
        srcPos_mm=(0.0, 0.0, 1000),
        detPos_mm=(0.0, 0.0, -500),
        prjSize_pixels=(2048, 2048),
        prjRes_mm=(0.388, 0.388),
        scan_velocity_degPersec_int=6,
        writeProjectionsInHD5File=False,
        projectionFileName="",
        debugFlag=False
        ):
    """
    vol_data_ZYX : numpy array, with depth Z (axial) as 1st dimension
    vol_res_XYZ_mm : numpy 1D array with resolution in mm in X(row), Y(column), Z(depth) dimension
    vol_pos_XYZ_mm : numpy 1D array describing  center of CT volume in patient co-ordinate system
    iso_pos_XYZ_mm : numpy 1D array describing  planned isocenter in  CT volume in patient co-ordinate system, the point that aligns with machine isocenter   
    alreadyNormalizedFlag: If not sure, mark as False
    hu_offset = hu_min: converting from normalized back to HU : -1000
    hu_max : max HU value = 400
    hu_slope: converting from normalized back to HU: 50000
    proj_angles_degrees: numpy 1D array with gantry angles in degrees at which projections will be  computed (0.0, 90.0) 
    srcPos_mm: tuple describing source position in IEC co-ordinate system : (0.0, 0.0, 1000)
    detPos_mm: tuple describing center of detector position in IEC co-ordinate system : (0.0, 0.0, -500)
    prjSize_pixels: tuple describing detector width and height in number of pixels : (2048, 2048)
    prjRes_mm: tuple describing detector pixel width and height in mm : (0.388, 0.388)
    scan_velocity_degPersec_int : integer: gantry speed in degrees per sec : 6
    writeProjectionsInHD5File : if we want to write out the projections in hd5 file
    projectionFileName: HD5 file name where projections will be written out.
    debugFlag : print debug information
    """
    import torch
    from viu.io import volume
    from viu.io import hdfproj
    from viu.torch.varian_reco_ext.varian_reco import VForwardprojectionModule
    from viu.util.cbct_geometry import CBCTGeometry

    volMax = np.max(vol_data_ZYX)
    volMin = np.min(vol_data_ZYX)
    if True==debugFlag:
        print(f'loaded volume min {volMin} max {volMax}')
    #First scale volume to 0-1 (if not already)
    volScaled_data_ZYX = (vol_data_ZYX - volMin) / (volMax - volMin)
    #Then scale to hu_min to hu_max
    hu_min=hu_offset
    vol_ZYX_HU = hu_min + volScaled_data_ZYX * (hu_max -hu_min)

    num_proj=len(proj_angles_degrees)
    if True==debugFlag:
        print(f'vol_ZYX_HU minHU {np.min(vol_ZYX_HU)} maxHU {np.max(vol_ZYX_HU)} shape {vol_ZYX_HU.shape} res_XYZ {vol_res_XYZ_mm} vol-center pos_XYZ {vol_pos_XYZ_mm} isocenter pos_XYZ {iso_pos_XYZ_mm} ')
        print(f'num_proj {num_proj} angles_degrees {proj_angles_degrees}')

    #Create geometry for forward projection
    geo = CBCTGeometry()
    # geo.fromTruebeam(half_fan=False, filtered=True)
    geo.fromTruebeam(half_fan=False, filtered=False)#<--- Not really needed
    # geo.print()
    # 256*2mm=512mm at isocenter
    # 512*1500/1000 = 768mm projected at isocenter
    # 1024*0.388 = 397.312mm imager size in mm << 768mm
    # 768 *0.388 = 297.984mm imager size in mm << 768mm
    geo.fromPos(srcPos=srcPos_mm,
                detPos=detPos_mm,
                prjSize=prjSize_pixels,
                prjRes=prjRes_mm,
                volSize=vol_ZYX_HU.shape,
                volRes=vol_res_XYZ_mm,
                scanVelocity=scan_velocity_degPersec_int)
    geo.check()
    if True==debugFlag:
        geo.print()

    # prj_cnt = len(angles)
    # print(f'FP of input volume: Shape_ZYX {np.shape(vol)} Resolution_XYZ{vol_res} Position_XYZ {vol_pos}')
    #convert to attenuation
    vol_ZYX_attenuation = volume.convert_unit(vol_ZYX_HU, unit = 'HU', to_unit='Attenuation', offset=hu_offset, slope = hu_slope)
    prj_stack_shape = [num_proj, geo.prjSize[1], geo.prjSize[0]]
    device = torch.device("cuda:0")
    vol_XYZ_shape = vol_ZYX_attenuation.shape[::-1] #Shape ZYX to shape XYZ
    vol_tr = vol_pos_XYZ_mm - iso_pos_XYZ_mm #The vector from planning isocenter to the CT volume center, this helps to identify each CT voxel in IEC co-ordinate system
    fpm = VForwardprojectionModule(
        prj_stack_shape=prj_stack_shape,
        prj_res=geo.prjRes,
        vol_size=vol_XYZ_shape,
        vol_res=vol_res_XYZ_mm,
        vol_tr=vol_tr,
        sad=geo.sad,
        use_half_fan_weighting=geo.half_fan).to(device, torch.float32)
    iso2pix = torch.tensor(geo.createIso2Pix(angles=proj_angles_degrees)[np.newaxis, ...]).cuda()
    #print(iso2pix)
    fpm.set_iso2pix(iso2pix, prj_cnt=num_proj)
    vol = torch.tensor(vol_ZYX_attenuation[None, ...]).cuda()
    prjs = fpm(vol)
    projs = prjs.detach().squeeze().cpu().numpy()
    if True==writeProjectionsInHD5File:
        print(f'Wring projections in {projectionFileName}')
        #In current version of VImaging Utils, default parameter additional_acq_params =None will give  hdf file creation error
        #Therefore we create dummy additional_acq_params dictionary and pass it while file creation
        dummy_additional_acq_params_dict = {'ImagerOrigin':'LT', 'additional_acq_params': 'dummy'}
        hdfproj.write_proj(projectionFileName, projs, iso2pix.detach().squeeze().cpu().numpy(), geo.prjRes,
            proj_angles_degrees, geo.sad, geo.sid, geo.imagerLat, geo.half_fan, geo.scanVelocity, overwrite=True,
            additional_acq_params=dummy_additional_acq_params_dict)
    return projs