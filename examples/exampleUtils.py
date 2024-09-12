#Imports
import os, sys, json, pathlib, shutil, glob
import argparse
import pandas as pd
import csv
import SimpleITK as sitk
import nibabel as nib
import random
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import scipy
from scipy.io import loadmat
from scipy import  signal

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import ipywidgets as widgets
from ipywidgets import interactive,interact, interact_manual, HBox, Layout,VBox
from IPython.display import display, clear_output

import interpol
from interpol.api import affine_grid

from scipy import ndimage

from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from viu.io import volume
from viu.io.volume import read_volume
from viu.torch.deformation.fields import DVF, set_identity_mapping_cache
from viu.torch.io.deformation import *
from viu.util.body_mask import seg_body

# import ipywidgets as ipyw
import ipywidgets as widgets
from ipywidgets import interactive,interact, interact_manual, HBox, Layout,VBox
from IPython.display import display, clear_output

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

# Rotation around Dicom Z axis : As viewed into Axial plane    from origin toward +Z axis (towards H(Head)), with A (Anterior) on top of the view, pass clockwise rotation is positive
# Rotation around Dicom Y axis : As viewed into Coronal plane  from origin toward +Y axis (towards P(Posterior)), with H (Head) on top of the view, pass clockwise rotation is positive
# Rotation around Dicom X axis : As viewed  (lying down horizontally) into Sagittal plane from +X toward origin (towards R(Right)) with H (Head) on top of the view, pass clockwise rotation is positive

def getPushRotationMatrix(theta_deg, viewString, center_slice_z, center_row_y, center_col_x):
    """
    viewString : axial, coronal, sagittal
    center_slice_z : center along 1st dimension
    center_row_y : center along 2nd dimensioon
    center_col_x : center along 3rd dimension
    """
    assert viewString in ["axial", "coronal", "sagittal"], f'viewString {viewString} not in ["axial", "coronal", "sagittal"]'
    rad = np.radians(theta_deg)
    if "axial"==viewString:
        rot =           np.array([[1., 0, 0, 0],[ 0, np.cos(rad), np.sin(rad), 0],[0, -np.sin(rad), np.cos(rad), 0],[0, 0, 0, 1.] ], 'float32')
    if "coronal"==viewString:
        rad = -rad
        rot =           np.array([[ np.cos(rad), 0, np.sin(rad), 0],[0, 1., 0, 0],[ -np.sin(rad), 0, np.cos(rad), 0],[0, 0, 0, 1.] ], 'float32')
    if "sagittal"==viewString:
        rad = -rad
        rot =           np.array([[ np.cos(rad), np.sin(rad), 0, 0],[-np.sin(rad), np.cos(rad), 0,  0],[0, 0, 1., 0],[0, 0, 0, 1.] ], 'float32')
    # print(f'rot: {rot}')
    transToOrigin = np.array([[1, 0, 0, -center_slice_z],[ 0, 1, 0, -center_row_y],[0, 0, 1, -center_col_x], [0, 0, 0, 1.]], 'float32')
    # print(f'transToOrigin: {transToOrigin}')
    transToCenter = np.array([[1, 0, 0, center_slice_z],[ 0, 1, 0, center_row_y],[0, 0, 1, center_col_x], [0, 0, 0, 1.]], 'float32')
    # print(f'transToCenter: {transToCenter}')
    pushAffine_np= transToCenter @ rot @ transToOrigin
    return pushAffine_np

#Generate normalized torch-functional  grid   from  torch-interpol grid.
def convertGrid_interpol2functional(interpol_grid_batched, depth, height, width):
    # printTensor("interpol_grid_batched", interpol_grid_batched)
    batchSize=interpol_grid_batched.shape[0]
    #normalization matrix
    normalizationMat = torch.tensor([[2./depth, 0, 0, -1.],[ 0, 2./height, 0, -1.],[0, 0, 2./width, -1.], [0, 0, 0, 1.]],
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
    field_ij_batched_normalized = torch.matmul(normalizationMat_rot, interpol_grid_batched.unsqueeze(-1)).squeeze(-1) + normalizationMat_tr
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
    deNormalizationMat = torch.linalg.inv(torch.tensor([[2./depth, 0, 0, -1.],[ 0, 2./height, 0, -1.],[0, 0, 2./width, -1.], [0, 0, 0, 1.]],
        dtype=torch.float32, device=functional_grid_batched.device))
    nb_dim = deNormalizationMat.shape[-1] - 1
    deNormalizationMat_rot = deNormalizationMat[:nb_dim, :nb_dim]
    deNormalizationMat_tr = deNormalizationMat[:nb_dim, -1]
    #Expand deNormalization matrix by batchSize
    deNormalizationMat_rot = deNormalizationMat_rot.expand(batchSize, *deNormalizationMat_rot.shape)
    deNormalizationMat_tr =   deNormalizationMat_tr.expand(batchSize, *deNormalizationMat_tr.shape)
    # Add dimension (in-place) in the end to support  matmul with normalizationMat_rot.
    # Then remove that dimension before adding  with normalizationMat_tr
    field_ij_batched_deormalized = torch.matmul(deNormalizationMat_rot, field_ij_batched_normalized.unsqueeze(-1)).squeeze(-1) + deNormalizationMat_tr
    return field_ij_batched_deormalized

def getPyTorchAffineMatTensor(unNormalizedAffineMatImageCoord_a2b_np,  depth_a, height_a, width_a,  depth_b, height_b, width_b, device=torch.device('cpu')):
    """
    parameters:
    unNormalizedAffineMatImageCoord_a2b_np : 4x4 (homogeneous) affine matrix, used in torch-interpol,  to be multiplied with pixel location (row, col) = (i,j) of location
    in volume a to obtain location in volume b,
    depth_a, height_a, width_a: depth, height, width  of vol a
    depth_b, height_b, width_b:  depth, height, width  of vol b
    returnPyTorchTheta: If true, it also returns pyTorch Theta (to be used by F.affine_grid or VIU DVF.affine).

    It should be noted that pyTorch Theta is a tensor version of  normalizedAffineMat but to be applied on normalised (x,y)
    instead of un-normalised pixel co-ordinate (i,j) as is the norm in  torch-interpol. Therefore, after normalization the rows of the matrix are swapped.

    Finally the last homogeneous row is not required in 

    return:
    1. normalized affine matrix  that is to be multiplied with mormalized pixel location (y,x) (from -1 to +1) of image a to obtain
    normalized pixel location in image b.
    2. pyTorch Theta to be used by F.affine_grid if returnPyTorchTheta is  true

    """
    T_normalized2Regular_a_yx = np.linalg.inv(np.array([[2./depth_a, 0, 0, -1.],[ 0, 2./height_a, 0, -1.],[0, 0, 2./width_a, -1.], [0, 0, 0, 1.]], 'float32'))
    T_regular2Normalized_b_yx =               np.array([[2./depth_b, 0, 0, -1.],[ 0, 2./height_b, 0, -1.],[0, 0, 2./width_b, -1.], [0, 0, 0, 1.]], 'float32')
    #Normalize, convert into tensor, use 1st 3 rows.
    pyTorchAffineMatTensor = torch.from_numpy(T_regular2Normalized_b_yx @ unNormalizedAffineMatImageCoord_a2b_np @ T_normalized2Regular_a_yx).to(device)[0:3,:]
    # In rows and columens: 0->2, 2->0, 1->1 [0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1],[2,2]
    block3x3Flipped  =  pyTorchAffineMatTensor[0:3, 0:3].flip([0,1])
    pyTorchAffineMatTensor[0:3, 0:3]=block3x3Flipped
    return pyTorchAffineMatTensor

def getUnNormalizedAffineMatTensorInImageCoord(pyTorchAffineMatTensorInNormalizedCoord_a2b,  depth_a, height_a, width_a,  depth_b, height_b, width_b):
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
    tmpAffineMat=pyTorchAffineMatTensorInNormalizedCoord_a2b.clone()
    tmpAffineMat[0:3, 0:3]=block3x3Flipped

    T_regular2normalized_a_yx = torch.tensor([[2./depth_a, 0, 0, -1.],[ 0, 2./height_a, 0, -1.],[0, 0, 2./width_a, -1.], [0, 0, 0, 1.]], dtype=torch.float32).to(device)
    T_normalized2regular_b_yx = torch.tensor([[2./depth_b, 0, 0, -1.],[ 0, 2./height_b, 0, -1.],[0, 0, 2./width_b, -1.], [0, 0, 0, 1.]], dtype=torch.float32).to(device).inverse()    #Normalize, convert into tensor, use 1st 3 rows.
    unNormalizedAffineMatTensorInImageCoord = T_normalized2regular_b_yx.matmul(tmpAffineMat.matmul(T_regular2normalized_a_yx))
    return unNormalizedAffineMatTensorInImageCoord