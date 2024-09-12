# %%
#Adapted from   /mnt/data/supratik/demonstrateDIR/v2_evalDIRParamDataGeneration.py
import argparse
from argparse import Namespace
import os
import re
from functools import reduce
from glob import glob
from time import sleep
import json
import csv

import imageio #Bose: Imageio is a Python library that provides an easy interface to read and write a wide range of image data, including animated images, volumetric data
import matplotlib.cm as cm #Bose: matplotlib colormaps and functions
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize #Bose: The matplotlib.colors module is used for converting color or numbers arguments to RGBA or RGB.This module is used for mapping numbers to colors or color specification conversion in a 1-D array of colors also known as colormap.And Normalize class is used to normalize data into the interval of [0.0, 1.0].
from skimage.segmentation import mark_boundaries #Bose: Return image with boundaries between labeled regions highlighted
from skimage.transform import rescale #Bose: Rescale operation resizes an image by a given scaling factor. The scaling factor can either be a single floating point value, or multiple values - one along each axis.
colormap = cm.hsv
norm = Normalize()

import pydicom
from scipy.ndimage import morphology
from torch.nn import MSELoss
from viu.io import volume
from viu.io.volume import read_volume
from viu.torch.deformation.fields import DVF, set_identity_mapping_cache
from viu.torch.io.deformation import *
from viu.util.body_mask import seg_body
import nibabel as nib
from viu.util.memory import fmt_mem_size
from viu.util.config import json_config

from pamomo.pca.cmo_pca import CMoPCA
from pamomo.registration.deformable import reg, force_unload
from viu.torch.visualization.ortho_utils import save_ortho_views #from pamomo.visualization.ortho_utils import save_ortho_views
from pamomo.visualization.cmo_pca_plots import *

import matplotlib.pyplot as plt

from viu.torch.measure.voi import measure_voi_list
from pamomo.metrices.residual_deformation import *


# %%
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


# %%
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


# %%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# %%
def generateGifFile(patientParentFolder, patientMRN, behaviourPrefixedConfigKey, vols, diff_vols, res, pos, fps, logFilepath):
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
        pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view.png'
        xView_pngFileName = f'{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_xView.png'
        pngFilePath = os.path.join(gifInputOutputFolder,pngFileName)
        xView_pngFilePath = os.path.join(gifInputOutputFolder,xView_pngFileName)
        save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', vols[phaseIdx,...], res, pos,
                                dst_path=gifInputOutputFolder, fn=pngFileName, views=cfg.views, additionalSingleViewToSave='x',additional_fn=xView_pngFileName)
        listOfFrameFilePaths.append(pngFilePath)
        xView_listOfFrameFilePaths.append(xView_pngFilePath)

        diff_pngFileName = f'diff_{behaviourPrefixedConfigKey}_{patientMRN}_phase_{phaseIdx:02d}_view.png'
        diff_pngFilePath = os.path.join(gifInputOutputFolder,diff_pngFileName)
        save_ortho_views(f'{phaseIdx}:{behaviourPrefixedConfigKey}', diff_vols[phaseIdx,...], res, pos,
                                dst_path=gifInputOutputFolder, fn=diff_pngFileName, views=cfg.views)
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

# %%
workingFolderParent = '/home/wd974888/Downloads'
logFilepath = f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/log.txt'
patiendId = 1 #in range(8,9): #range(1,12) #range(1,12) #Use truncateDepth_initial=75, truncateDepth_final=50 for Patient=01 and 0,0 for Patient=04
binningType = 'PB' # in ['AB']: #['AB', 'PB']
patientMRN = f'Patient{patiendId:02d}{binningType}'
print(f'patientMRN {patientMRN}')
args =  Namespace(\
    vol=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/{patientMRN}/StudyAnonymized/bin*',\
    pca_fn=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/{patientMRN}/{patientMRN}_test_pca.hdf',\
    dvfs=None,\
    cache_dvf_npz_folder=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/{patientMRN}/DVFStore/',\
    vols_fn=f'{patientMRN}_test_vols.hdf',\
    cache=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/{patientMRN}/Cache/',\
    body_seg=True,\
    body_seg_union=True,\
    nifti_body_seg='seg/body.nii.gz',\
    air_threshold=-300,\
    skip=1,\
    skip_idx=None,\
    deferred_resampling=True,\
    force_pca=True,\
    plot=True,\
    reconstructedVol_fn=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/{patientMRN}/{patientMRN}_recontructed_vols.hdf',\
    mean2dcm=True,\
    dirOptionsDict_fn=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/dirOptionsDict.json',\
    fps=4,\
    prefilter=False,\
    truncateDepth_initial=75,\
    truncateDepth_final=50\
    )
print(args)

# %%
mse = MSELoss()

# %%
if args.dvfs is  None: #No Dicom DVF folder  specified
    if args.cache_dvf_npz_folder is not None: # cache dvf_npz_folder is specified
        #Create folder if not already present
        os.makedirs(args.cache_dvf_npz_folder, exist_ok=True)

# %%
with open(str(args.dirOptionsDict_fn),"r") as json_file:
    dirOptionsDict = json.load(json_file)

# %%
usingLoopNoForceUnload=True
#Before loop over configuration:
args.org_pca_fn = args.pca_fn
args.org_reconstructedVol_fn = args.reconstructedVol_fn
prefilterPrefix = "Interpol" if True==args.prefilter else "Functional"

# %%
currentDirOptionsKey = f"config10" #in [f"config{i:02d}" for i in range(10,11)]: #[f"config{i:02d}" for i in range(1,11)]
behaviourPrefixedConfigKey = f'{prefilterPrefix}_{currentDirOptionsKey}'
#Check if the resultGifFilePath already exists. If exists, continue
gifFilePath = os.path.join(f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/gifFolder/',f'{patientMRN}_fps_{args.fps:02d}_{behaviourPrefixedConfigKey}.gif') #{currentDirOptionsKey}.gif
gifFilePathOrgVol = os.path.join(f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/gifFolder/',f'{patientMRN}_fps_{args.fps:02d}_org.gif')
if ("config01"==currentDirOptionsKey and os.path.exists(gifFilePathOrgVol) and  os.path.exists(gifFilePath))\
    or ("config01"!=currentDirOptionsKey and  os.path.exists(gifFilePath)):
    resultForThisConfigExists=True
    print(f'Exists {gifFilePath}. Moving to next item')
else:
    resultForThisConfigExists=False
    print(f'DOES NOT EXISTS {gifFilePath}. Will generate gif')

#Overwrite previous decision
runGifCreationFlag=True

# %%
# if False==resultForThisConfigExists:
######## LOG ######
logString = f'Started working on {behaviourPrefixedConfigKey}_{patientMRN}'
print(logString)
with open(logFilepath, 'r+') as f:
    f.seek(0)
    f.writelines([logString])
    f.truncate()
    f.close()
###################

# %%
# if False==resultForThisConfigExists:
currentDirOptions = dirOptionsDict[currentDirOptionsKey]
#Update args to include configuration
dirName = os.path.dirname(args.org_pca_fn)
f_name = os.path.basename(args.org_pca_fn).split('.')[0]
f_extension = os.path.basename(args.org_pca_fn).split('.')[1]
configurized_pca_fn = os.path.join(dirName, f'{behaviourPrefixedConfigKey}_{f_name}.{f_extension}')
args.pca_fn = configurized_pca_fn
####
args.reconstructedVol_fn = os.path.join(os.path.dirname(args.org_reconstructedVol_fn), f'{behaviourPrefixedConfigKey}_{os.path.basename(args.org_reconstructedVol_fn)}')
###
############
# #Inconsistency check #1. Move here after pca_fn is prepended with configName
# if not args.force_pca and os.path.exists(args.pca_fn):
#     exit()
# ############
dst_path = os.path.realpath(dirName)
os.makedirs(dst_path, exist_ok=True)
name=f'{behaviourPrefixedConfigKey}_{f_name}'
vols = None
dvfs = None
msks = None
if args.cache is not None:
    cache_fn = os.path.join(args.cache, os.path.splitext(os.path.basename(args.pca_fn))[0] + '.pt')
    #If cached filename is present, read vols, dvfs, masks from cached file
    if os.path.exists(cache_fn):
        print(f'Cache file exists: {cache_fn}')
        vols, dvfs, res, pos, dvf_res, dvf_pos, msks, vol_idx_msk = torch.load(cache_fn)
        #vols = (10, 590, 512, 512), dvfs = (10, 590, 512, 512, 3) msks(body union) = (1, 590, 512, 512)
    else:
        print(f'Cache file does NOT exist: {cache_fn}')
dst_path_org=dst_path

# %%
# if False==resultForThisConfigExists:
#We are here if  cached filename is NOT present
#First read the volumes
if vols is None:
    vol_fnl = glob(args.vol)
    vol_fnl = sort_by_series_number(vol_fnl)

    vol_idx_msk = torch.ones(len(vol_fnl), dtype=bool)
    if args.skip > 0:
        vol_idx_msk[torch.arange(len(vol_idx_msk)) % (args.skip + 1) != 0] = False

    if args.skip_idx is not None:
        for i in args.skip_idx:
            vol_idx_msk[i] = False

    vol_fnl = [vol_fnl[i] for i, b in enumerate(vol_idx_msk) if b]

    #########
    tmp_vol, _, _ = read_volume(vol_fnl[0])
    org_depth=tmp_vol.shape[0]
    startSlice=args.truncateDepth_initial
    endSlice=org_depth - args.truncateDepth_final
    new_depth = org_depth - args.truncateDepth_initial - args.truncateDepth_final
    print(f'org_depth {org_depth} startSlice {startSlice} endSlice {endSlice} new_depth {new_depth}')
    #########
    vol_lst = []
    for vfn in vol_fnl:
        print(f'Loading {vfn}...')
        vol, res, pos = read_volume(vfn)
        ######
        vol=vol[startSlice:endSlice, ...]
        ######
        vol_lst.append(torch.Tensor(vol)[None, ...])
    vols = torch.concatenate(vol_lst, dim=0)
    del vol_lst

# %%
# Let us display one bin
from exampleUtils import *
display_vol = vols[0,...].cpu().numpy()
v1_volumeComparisonViewer3D(
    listVolumes=[display_vol],listLabels=[f'{patientMRN}_bin00'],
    maxZ0=display_vol.shape[0], maxZ1=display_vol.shape[1], maxZ2=display_vol.shape[2],
    figsize=(12,8), cmap='gray',
    displayColorbar=False, useExternalWindowCenter=True, wMin=-500, wMax=500, useAspectCol=False)


# %%
# if False==resultForThisConfigExists:
#We are here if  cached filename is NOT present
#Now read / compute DVFs
if dvfs is None:
    dvf_lst = []
    #If DICOM DVF filename list is available, the DVFs are read from there.
    if args.dvfs is not None:
        dvf_fnl = glob(args.dvfs)
        for dfn in dvf_fnl:
            print(f'Loading {dfn}...')
            dvf, dvf_res, dvf_pos = read_nr_dcm(dfn, vol_dim=vol.shape[::-1], vol_res=res, vol_pos=pos)
            dvf_lst.append(DVF(torch.Tensor(dvf)[None, ...]))
    #Otherwise  look for DVFs from npz file cache if specified or run DIR
    else:
        #Fist though get the body mask of each scans and if needed make
        vol_min = -1200
        cnt = vols.shape[0]
        dst_dim = torch.tensor(vols[0].squeeze().shape[::-1])
        dst_res = torch.tensor(res, dtype=torch.float64)
        msk_lst = []
        if args.body_seg:
            for i in range(cnt):
                #Mask is always obtained on the untruncated volume and then truncated as trying to obtain mask on truncated volume is giving empty mask!!!
                if args.nifti_body_seg is not None:
                    fn = os.path.join(vol_fnl[i], args.nifti_body_seg)
                    print(f'Load body segmentation: {fn}...')
                    ####
                    # msk = torch.tensor(nib.load(fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1)[None, ...]
                    # Truncate as needed
                    msk = torch.tensor(nib.load(fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1)[startSlice:endSlice, ...][None, ...]
                    ####
                else:
                    print('Find connected components...')
                    msk = torch.tensor(seg_body(vols[(i + 1) % cnt], args.air_threshold, air_dilation=4))
                print(f'msk shape {msk.shape}')
                msk_lst.append(msk)
            msks = torch.concatenate(msk_lst, dim=0)
            del msk_lst

            if args.body_seg_union:
                msks = reduce(lambda a, b: a.bitwise_or(b), msks)[None, ...]
                # hack to fill disconnected air components at the volume borders
                from skimage import measure
                label = measure.label(torch.bitwise_not(msks).numpy())
                idx, label_cnt = np.unique(label, return_counts=True)
                idx = idx[np.argsort(label_cnt)[:-1]]
                for i in idx:
                    #Exception in PyTorch 2.2 env: IndexError: The shape of the mask [1, 590, 512, 512] at index 0 does not match the shape of the indexed tensor [590, 512, 512] at index 0
                    #Because msks[0].shape = [590, 512, 512], But [label == i].shape = [1, 590, 512, 512]
                    msk[0][label == i] = True #Replaced [0] to remove exception in Baden environment #msks[0][label == i] = True
                # ---

        co_dvf_res = None
        co_dvf_pos = None
        #Now compute cyclic DVFs.
        for i in range(cnt):
            #However before running DIR check whether it has been pre-computed and saved as npz file
            dvfFileName=''
            if (args.cache_dvf_npz_folder is not None):
                from_scanName = os.path.basename(vol_fnl[i])
                to_scanName = os.path.basename(vol_fnl[(i + 1) % cnt])
                #NOTE dvfs themselves are initially generated from MEVIS and do not depend on functiona / interpol and hence 
                # instead of behaviourPrefixedConfigKey, they are named with just currentDirOptionsKey
                dvfFileName = os.path.join(args.cache_dvf_npz_folder, f'{currentDirOptionsKey}_dvf_from_{from_scanName}_to_{to_scanName}.npz')

            if(os.path.exists(dvfFileName)):
                dvfBatch = np.load(dvfFileName)
                dvf = dvfBatch['arr_0']
                dvf_res = dvfBatch['arr_1']
                dvf_pos =  dvfBatch['arr_2']
            else:
                src_seg_dict = {}
                dst_seg_dict  = {}

                src_seg_list = currentDirOptions["src_seg"]
                dst_seg_list = currentDirOptions["dst_seg"]
                alpha_DIR = currentDirOptions["alpha"]
                beta_DIR = currentDirOptions["beta"]
                gamma_DIR = currentDirOptions["gamma"]
                dvfRes = currentDirOptions["dvfRes"]
                prefix = currentDirOptions["prefix"]
                if"default"==dvfRes:
                    numLevels=3 #=default value =3
                    finestLevelReference=1 #=default value =1 => DVF computed at half resolution of volume
                    finestLevelTemplate=1  # =default value = 1 => DVF computed at half resolution of volume
                elif"low"==dvfRes:
                    numLevels=3
                    finestLevelReference=2
                    finestLevelTemplate=2
                elif"high"==dvfRes:
                    numLevels=4
                    finestLevelReference=0
                    finestLevelTemplate=0
                else:
                    print('Unacceptable dvf resolution.')
                    exit()

                if "SG_lungs" in src_seg_list and "SG_lungs" in dst_seg_list:
                    src_lungs_fn = os.path.join(vol_fnl[i], 'seg/lung.nii.gz')
                    src_lungs_np = torch.tensor(nib.load(src_lungs_fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1).numpy() #np.transpose(nib.load(src_lungs_fn).get_fdata(), (2,1,0)).astype('uint8').astype('bool')
                    src_seg_dict["SG_lungs"]=src_lungs_np
                    dst_lungs_fn = os.path.join(vol_fnl[(i + 1) % cnt], 'seg/lung.nii.gz')
                    dst_lungs_np = torch.tensor(nib.load(dst_lungs_fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1).numpy() #np.transpose(nib.load(dst_lungs_fn).get_fdata(), (2,1,0)).astype('uint8').astype('bool')
                    dst_seg_dict["SG_lungs"]=dst_lungs_np

                if "LR_dst_bones" in dst_seg_list:
                    dst_bones_fn = os.path.join(vol_fnl[(i + 1) % cnt], 'seg/bone.nii.gz')
                    dst_bones_np = torch.tensor(nib.load(dst_bones_fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1).numpy() #np.transpose(nib.load(dst_bones_fn).get_fdata(), (2,1,0)).astype('uint8').astype('bool')
                    dst_seg_dict["LR_dst_bones"]=dst_bones_np

                if "LR_dst_vertebra" in dst_seg_list:
                    dst_vertebra_fn = os.path.join(vol_fnl[(i + 1) % cnt], 'seg/vertebra_combined.nii.gz')
                    dst_vertebra_np = torch.tensor(nib.load(dst_vertebra_fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1).numpy() #np.transpose(nib.load(dst_bones_fn).get_fdata(), (2,1,0)).astype('uint8').astype('bool')
                    dst_seg_dict["LR_dst_vertebra"]=dst_vertebra_np

                if "CJ_dst_lungs" in dst_seg_list:
                    dst_lungs_fn = os.path.join(vol_fnl[(i + 1) % cnt], 'seg/lung.nii.gz')
                    dst_lungs_np = torch.tensor(nib.load(dst_lungs_fn).get_fdata()).to(bool).permute(2, 1, 0).flip(1).numpy() #np.transpose(nib.load(dst_lungs_fn).get_fdata(), (2,1,0)).astype('uint8').astype('bool')
                    dst_seg_dict["CJ_dst_lungs"]=dst_lungs_np

                if args.body_seg:
                    dst_seg_dict["SM_bodymask"]=msks[min(i, msks.shape[0]-1)].numpy()

                additional_args = {}
                if len(src_seg_dict) > 0:
                    additional_args['src_seg'] = src_seg_dict
                if len(dst_seg_dict) > 0:
                    additional_args['dst_seg'] = dst_seg_dict
                if args.body_seg:
                    additional_args['similarityMaskMultilevelStrategy'] = 'STRICTINTERIOR'

                print('Start registration...')
                if"default"==dvfRes:
                    dvf, dvf_res, dvf_pos = reg(vols[i].clip(min=vol_min).squeeze().numpy(), res,
                        vols[(i + 1) % cnt].clip(min=vol_min).squeeze().numpy(), res,
                        alpha=alpha_DIR, #20
                        maskAlignmentWeight=beta_DIR,
                        # numLevels=3, #=default value =3
                        # finestLevelReference=1, #=default value =1 => DVF computed at half resolution of volume
                        # finestLevelTemplate=1,  # =default value = 1 => DVF computed at half resolution of volume
                        constantJacobianWeight=gamma_DIR,
                        # verboseMode='false',
                        **additional_args)
                else:
                    dvf, dvf_res, dvf_pos = reg(vols[i].clip(min=vol_min).squeeze().numpy(), res,
                        vols[(i + 1) % cnt].clip(min=vol_min).squeeze().numpy(), res,
                        alpha=alpha_DIR, #20
                        maskAlignmentWeight=beta_DIR,
                        numLevels=numLevels,
                        finestLevelReference=finestLevelReference,
                        finestLevelTemplate=finestLevelTemplate,
                        constantJacobianWeight=gamma_DIR,
                        # verboseMode='false',
                        **additional_args)

                #Save dvf if we have run DIR and dvfFileName is not ''
                if (dvfFileName !='' and False==os.path.exists(dvfFileName)):
                    np.savez_compressed(dvfFileName, dvf, dvf_res, dvf_pos)

            dvf_res = torch.tensor(dvf_res, dtype=torch.float64)
            dvf_pos = torch.tensor(dvf_pos, dtype=torch.float64)
            #dvf.shape before resampling (295, 256, 256, 3)
            dvf = DVF(dvf[None,...]).from_millimeter(dvf_res).to(torch.float32) #Adding batch before constructing DVF object
            if not args.deferred_resampling:
                dvf = dvf.resample(dst_dim, dst_res, dvf_res=dvf_res, dvf_pos=dvf_pos, prefilter=args.prefilter, mode='cubic' if True==args.prefilter else 'bilinear') #TODO
            else:
                if co_dvf_res is None:
                    co_dvf_res, co_dvf_pos = dvf_res, dvf_pos
                else:
                    assert torch.equal(dvf_res, co_dvf_res)
                    assert torch.equal(dvf_pos, co_dvf_pos)

            dvf_lst.append(dvf[None, ...])# In new build_cyclic_pca.py, with bi_cycle=False, this line is equivalent to dvf_lst.append(cc_dvf[None, None, ...])

    dvfs = torch.concatenate(dvf_lst, dim=1) ## In new build_cyclic_pca.py, with bi_cycle=False,  #With deferred resampling, dvfs.shape= torch.Size([1, 10, 105, 256, 256, 3])
    del dvf_lst

    # Moved before memory clean-up
    if args.cache is not None:
        os.makedirs(os.path.realpath(os.path.dirname(cache_fn)), exist_ok=True)
        torch.save((vols, dvfs, res, pos, dvf_res, dvf_pos, msks, vol_idx_msk), cache_fn)# torch.save((vols, dvfs, res, pos, dvf_res, dvf_pos, msks), cache_fn)

    if False==usingLoopNoForceUnload:
        ##########################
        force_unload() #<-------- This is destroying the global variable _dir_lab so in the next loop of the for blocck registration is caching,  
        #                # Will see if invoking _dir_lib  from pamomo.registration.deformable fixes that.
        #                # Or may be move it out of the loop? or not call?
        free_mem, total_mem = 0, 1
        while free_mem / total_mem < 0.98:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f'Memory free: {fmt_mem_size(free_mem)}, total: {fmt_mem_size(total_mem)} {free_mem / total_mem:.2%}')
            sleep(1)
        ##########################

# %%
# if False==resultForThisConfigExists:
if args.vols_fn is not None:
    fn = os.path.join(dst_path, args.vols_fn)
    if False==os.path.exists(fn):
        print(f'Writing {fn}...')
        with h5py.File(fn, 'w') as hdf:
            volume.add_hdf_volume(hdf, vols.cpu().numpy(), res, pos, hdf_ds_name='volumes')
            volume.add_hdf_volume(hdf, msks.cpu().numpy(), res, pos, hdf_ds_name='masks')
    else:
        print(f'Already exists {fn}...')

# %%
createOrOverwritePCAFlag=False
if False==os.path.exists(args.pca_fn):
    createOrOverwritePCAFlag=True
print(f'createOrOverwritePCAFlag {createOrOverwritePCAFlag}')


# %%
if True==createOrOverwritePCAFlag:
    print(f'Running PCA  operation')
    set_identity_mapping_cache(True)
    pca_msk = None
    if msks is not None:
        pca_msk = reduce(lambda a, b: a.bitwise_or(b), msks) #shape: D,H,W


    cnt = dvfs.shape[0] #dvfs.shape: (NBins, D, H, W, 3)

    pca = CMoPCA(example_msk=vol_idx_msk,prefilter=args.prefilter) #vol_idx_msk : [ True, False, Tue, false,... totalBins]
    assert dvfs.shape[0] == 1, f'With bi-cycle=False, dvfs.shape=[1, NBins, D*, H*, W*, 3]'
    star_dvfs = pca.cycle_to_star_dvfs(dvfs[0].cuda())
    # else:
    #     res_vols = pca.resample_vols(vols.cuda(), res, pos, dvfs.shape[-4:-1], dvf_res, dvf_pos) #vols.shape [NBin, D, H, W] dvfs.shape = [NBin, D', H', W', 3]

    #     star_dvfs, mse_lst = pca.bicycle_to_star_dvfs(dvfs.cuda(), res_vols)
    #     if args.plot:
    #         vmax = mse_lst.max()
    #         fig, ax = plt.subplots(2)
    #         ax[0].imshow(mse_lst[0], vmax=vmax)
    #         ax[1].imshow(mse_lst[1], vmax=vmax)
    #         fig.savefig(os.path.join(dst_path, f'{name}_bicycle_mse.png'))
    #         plt.close(fig)


    pca.from_star(vols, res, pos, star_dvfs, dvf_res, dvf_pos, body_msk=pca_msk)
    log, residuals = pca.reconstruct_mean(vols, iterations=1000)

    pca.write(args.pca_fn)

    if args.plot:
        fig, ax = plt.subplots()
        ax.set_title(f'{name} Convergence')
        ax.plot(log, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylabel('HU$^2$')
        ax2 = ax.twinx()
        ax2.plot(np.diff(log), color='lightblue')
        ax2.tick_params(axis='y', labelcolor='lightblue')
        ax2.set_ylabel('$\Delta$ HU$^2$')
        fig.tight_layout()
        fig.savefig(os.path.join(dst_path, f'{name}_conv.png'))
        # plt.show()
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_title(f'{name} Residual MSE')
        ax.bar(range(len(residuals)), residuals)
        ax.set_xlabel('Bin')
        ax.set_ylabel('HU$^2$')
        fig.tight_layout()
        fig.savefig(os.path.join(dst_path, f'{name}_residuals.png'))
        # plt.show()
        plt.close(fig)

else:
    #Read PCA file
    print(f'Reading pre-created PCA  operation')
    pca = CMoPCA(example_msk=vol_idx_msk,prefilter=args.prefilter)
    mean, res, pos = pca.read(args.pca_fn, device='cuda') #mean.shape torch.Size([590, 512, 512]) #NOTE read mean volume

# %%
if True==runGifCreationFlag:
    #Code added to build_cyclic_pca.py
    ######## LOG ######
    logString = f'Writing mean volume for  {behaviourPrefixedConfigKey}_{patientMRN}'
    print(logString)
    with open(logFilepath, 'r+') as f:
        f.seek(0)
        f.writelines([logString])
        f.truncate()
        f.close()
    ###################
    args.amplitude_gated = ("AB_" in f_name)
    dst_path_subdir = os.path.join(dst_path, f'{behaviourPrefixedConfigKey}_amp' if args.amplitude_gated else f'{behaviourPrefixedConfigKey}_phase')
    os.makedirs(dst_path_subdir, exist_ok=True)
    # mean, res, pos = pca.read(args.pca_fn, device='cuda') #mean.shape torch.Size([590, 512, 512]) #NOTE read mean volume
    mean, res, pos =  pca.mean_vol, pca.res, pca.pos
    print(f'mean vol shape and device {mean.shape} {mean.device} res {res} pos {pos}')
    #Write mean volume
    if args.mean2dcm:
        # volume.write_dcm(dst_path, 'mean_dcm', mean, res, pos)
        volume.write_dcm(dst_path_subdir, 'mean_dcm', mean.detach().cpu().numpy(), res, pos)

    #Compute reconstructed volumes
    ######## LOG ######
    logString = f'Creating and writing reconstructed phase volumes for  {behaviourPrefixedConfigKey}_{patientMRN}'
    print(logString)
    with open(logFilepath, 'r+') as f:
        f.seek(0)
        f.writelines([logString])
        f.truncate()
        f.close()
    ###################    vols = torch.tensor(vols, device=mean.device)
    reconstructed_vol_lst = []
    for i in range(vols.shape[0]):
        ref_vol = vols[i].to(mean.device)
        print(f'Index: {i} ref_vol shape and device {ref_vol.shape} {ref_vol.device}')
        ref_dvf = pca.dvf(pca.vt[i, :]).to(mean.device)
        print(f'Index: {i} ref_dvf shape and device {ref_dvf.shape} {ref_dvf.device}')
        reconstructedPhaseVol = DVF(ref_dvf[None,...]).sample(mean[None, None,...],prefilter=args.prefilter, mode='cubic' if True==args.prefilter else 'bilinear' ).squeeze().squeeze() #reconstructedPhaseVol = DVF(ref_dvf)(mean)
        print(f'Index: {i} reconstructedPhaseVol shape and device {reconstructedPhaseVol.shape}  {reconstructedPhaseVol.device} ')
        reconstructed_vol_lst.append(reconstructedPhaseVol[None, ...].cpu().numpy())
    reconstructed_vols = np.concatenate(reconstructed_vol_lst, axis=0)
    with h5py.File(args.reconstructedVol_fn, 'w') as hdf:
        volume.add_hdf_volume(hdf, reconstructed_vols, res, pos, hdf_ds_name='volumes')
        print(f'Finished writing {args.reconstructedVol_fn}')
    ####### Tensorify and transfer to GPU before gif generation
    vols=vols.to(mean.device)
    reconstructed_vols=torch.tensor(reconstructed_vols, device=mean.device)
    del reconstructed_vol_lst
    del mean
    ###### Generate gif file for this config
    generateGifFile(patientParentFolder=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/',
        patientMRN=patientMRN, behaviourPrefixedConfigKey=behaviourPrefixedConfigKey, vols=reconstructed_vols, diff_vols= reconstructed_vols-vols, res=res, pos=pos, fps=args.fps, logFilepath=logFilepath)
    if "config01"==currentDirOptionsKey:
        #For the first configuration also generate gif file with original phase volumes
        generateGifFile(patientParentFolder=f'{workingFolderParent}/workingFolder/DeformationExperiment/PCA/',
            patientMRN=patientMRN, behaviourPrefixedConfigKey='org', vols=vols, diff_vols= vols-vols, res=res, pos=pos, fps=args.fps, logFilepath=logFilepath)

    #Memory clean up
    del reconstructed_vols

# %%
evalPCAFlag=False
if True==evalPCAFlag:
    args.dir=True
    args.dvf=True
    args.maximum_mag=None
    dst_path=dst_path_org
    print(f'original dst_path {dst_path}')
    args.amplitude_gated = ("AB_" in f_name)
    dst_path = os.path.join(dst_path, f'{behaviourPrefixedConfigKey}_amp' if args.amplitude_gated else f'{behaviourPrefixedConfigKey}_phase')
    print(f'new dst_path {dst_path}')
    os.makedirs(dst_path, exist_ok=True)
    stats_path = 'dvf_stats'
    os.makedirs(os.path.join(dst_path, stats_path), exist_ok=True)
    comp_dir = args.dir
    air_threshold = args.air_threshold #-300

    cfg = json_config(os.path.splitext(args.pca_fn)[0] + '.json')  #NOTE create json file for edge measurement
    if 'views' not in cfg.keys:
        cfg.add_config_item('views', [{'ctr': pos.tolist(), 'voi': None, 'wl': [500, 0]}])
        cfg.write()

    if 'edge_measurements' not in cfg.keys:
        cfg.add_config_item('edge_measurements', [])
        cfg.write()

    if 'maximum_mag' not in cfg.keys or args.maximum_mag is not None:
        cfg.add_config_item('maximum_mag', args.maximum_mag)
        cfg.write()

    dims = 3
    steps = [-3, -2, -1, 0, 1, 2, 3]
    for i in range(dims):
        a = torch.zeros(dims, device=pca.mean_vol.device)  #Should this be outside for i in range(dims) loop?
        for idx, wgt in enumerate(steps):
            a[i] = wgt / np.sqrt(pca.example_count) #a: vector of size dim, all values 0 except a[i]=steps[j]/sqrt(exampleCount)
            pca_vol = pca.vol(a) #pca.vol(a) = self.dvf(a)(self.mean_vol) 
            save_ortho_views(f'pc {i} sigma {wgt}', pca_vol, res, pos,
                            dst_path=os.path.join(dst_path, 'principal_components'),
                            fn=f'pc_{i:03}_{idx:03}_vol.png', views=cfg.views)

    A, p = pca.cyclic_parametrization(phase_gated=not args.amplitude_gated) #A.shape: [9,2] p.shape [2,10] 
    fig, ax = plot_params(p,pca.example_indices)
    fig.savefig(os.path.join(dst_path, 'parametrization.png'))
    plt.close(fig)

    vt_hat = (A @ p).transpose(1, 0) #[9,2]*[2,10]transpose(1, 0) =>  [10,9] : vt_hat.shape  : 10 volumes, 9 PC co-efficients /volume
    min_pc = 0.5
    pcc, relevant = pca.relevant_parameters(vt_hat, min_pc)
    pcc, relevant = pcc.cpu(), relevant.cpu() #pcc.shape [9] relevant.shape [9]
    relevant_cnt = int(relevant.sum()) #2

    fig, ax = plot_parameter_fit(pca.vt, vt_hat, pca.example_indices, pcc, relevant, min_pc)
    ax.set_xlabel(f'4D {"Amplitude Bin" if args.amplitude_gated else "Phase"}')
    fig.tight_layout()
    fig.savefig(os.path.join(dst_path, 'param_fitting.png'))
    plt.close(fig)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Standard Deviation (PCA)', color=color)
    ax1.plot(pca.std.cpu(), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('Correlation Coefficient (Parametrization)', color=color)  # we already handled the x-label with ax1
    ax2.plot(pcc.cpu(), 'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(dst_path, 'pca_std_pcc.png'))
    plt.close(fig)

    # vols_path = os.path.join(os.path.realpath(dirName), args.vols_fn)
    # args.vols_fn = vols_path
    # print(f'Loading {args.vols_fn}...')
    # vols, res, pos = volume.read_volume(args.vols_fn, hdf_ds_name='volumes')
    # vols = torch.tensor(vols, device='cuda')
    # msks, *_ = volume.read_volume(args.vols_fn, hdf_ds_name='masks')
    # if msks is not None:
    #     msks = torch.as_tensor(msks, dtype=bool)
    #####
    #vols, res, pos, msks are already defined
    ####
    vols=vols.to(pca.mean_vol.device)

    append_cnt = 0
    pca_mse_values = torch.zeros(vols.shape[0], pca.component_count+3)
    pca_msk_mse_values = torch.zeros(vols.shape[0], pca.component_count+3)
    dir_stats = np.empty((vols.shape[0], pca.component_count + 3), dtype=object)
    dvf_stats = np.empty((vols.shape[0], pca.component_count + 3), dtype=object)

    measurement_names = ['Reference', 'Mean'] + \
                        [f"#{j} PCs" for j in range(1, pca.component_count + 1)] + \
                        [f'Parametrization', f"Param. (#{relevant_cnt} PCs)"]
    measurement_list = []
    reconstructed_vol_lst = []
    for i in range(vols.shape[0]):
        ref_vol = vols[i]
        ref_dvf = pca.dvf(pca.vt[i, :]).cpu() #NOTE ref_dvf does not have batch dimension

        if msks is not None:
            ref_msk = msks[min(i, msks.shape[0] - 1)]
        else:
            ref_msk = seg_body(ref_vol.cpu().numpy(), air_threshold=air_threshold)
        ref_msk_np = ref_msk if isinstance(ref_msk, np.ndarray) else ref_msk.cpu().numpy()
        mvl = measure_voi_list(cfg.edge_measurements, ref_vol, res, pos)
        measurements = [m['dst'] for m in mvl]
        for j in range(pca.component_count + 1):
            pca_vol = pca.vol(pca.vt[i, :j])
            mvl = measure_voi_list(cfg.edge_measurements, pca_vol, res, pos)
            measurements += [m['dst'] for m in mvl]
            pca_mse_values[i, j] = mse(pca_vol, ref_vol).cpu()
            pca_msk_mse_values[i, j] = msk_mse(ref_vol, pca_vol, ref_msk)

            residual_dvf = (ref_dvf - pca.dvf(pca.vt[i, :j]).cpu()).millimeter(res)
            dvf_stats[i, j] = dvf_statistics(residual_dvf, ref_msk, max_value=cfg.maximum_mag)
            save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dvf_max_magnitude_{i}_{j}.png'),
                                f'Maximum Magnitude {measurement_names[j + 1]}', residual_dvf.magnitude(),
                                res, cfg.maximum_mag)

            if comp_dir:
                dvf = residual_deformation(ref_vol.cpu(), pca_vol.cpu(), res, ref_msk=ref_msk_np, prefilter=args.prefilter, mode='cubic' if True==args.prefilter else 'bilinear').squeeze() #residual_deformation(ref_vol.cpu(), pca_vol.cpu(), res, ref_msk=ref_msk)
                dir_stats[i, j] = dvf_statistics(dvf, ref_msk, max_value=cfg.maximum_mag)

                mag = dvf.magnitude()
                save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dir_max_magnitude_{i}_{j}.png'),
                                    f'Maximum Magnitude {measurement_names[j+1]}', mag, res, cfg.maximum_mag)


        pca_vol = pca.vol(pca.vt[i, :])
        #This is the reconstructed phase volume
        reconstructed_vol_lst.append(pca_vol[None, ...].cpu().numpy())
        vol_hat = pca.vol(vt_hat[i, :])
        vol_hat_relevant = pca.vol(vt_hat[i, :relevant_cnt])

        mvl = measure_voi_list(cfg.edge_measurements, vol_hat, res, pos)
        measurements += [m['dst'] for m in mvl]

        mvl = measure_voi_list(cfg.edge_measurements, vol_hat_relevant, res, pos)
        measurements += [m['dst'] for m in mvl]

        if len(measurements) > 0:
            measurement_list += [measurements]

        pca_mse_values[i, -2] = mse(vol_hat, ref_vol).cpu()
        pca_mse_values[i, -1] = mse(vol_hat_relevant, ref_vol).cpu()

        pca_msk_mse_values[i, -2] = msk_mse(ref_vol, vol_hat, ref_msk)
        pca_msk_mse_values[i, -1] = msk_mse(ref_vol, vol_hat_relevant, ref_msk)

        residual_dvf = (ref_dvf - pca.dvf(vt_hat[i, :]).cpu()).millimeter(res)
        dvf_stats[i, -2] = dvf_statistics(residual_dvf, ref_msk, max_value=cfg.maximum_mag)
        save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dvf_max_magnitude_{i}_param.png'),
                            f'Maximum Magnitude {measurement_names[-2]}', residual_dvf.magnitude(),
                            res, cfg.maximum_mag)

        residual_dvf = (ref_dvf - pca.dvf(vt_hat[i, :relevant_cnt]).cpu()).millimeter(res)
        dvf_stats[i, -1] = dvf_statistics(residual_dvf, ref_msk, max_value=cfg.maximum_mag)
        save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dvf_max_magnitude_{i}_param_{relevant_cnt}.png'),
                            f'Maximum Magnitude {measurement_names[-1]}', residual_dvf.magnitude(),
                            res, cfg.maximum_mag)

        if comp_dir:
            dvf = residual_deformation(ref_vol.cpu(), vol_hat.cpu(), res, ref_msk=ref_msk_np, prefilter=args.prefilter, mode='cubic' if True==args.prefilter else 'bilinear').squeeze() #dvf = residual_deformation(ref_vol.cpu(), vol_hat.cpu(), res, ref_msk=ref_msk)
            dir_stats[i, -2] = dvf_statistics(dvf, ref_msk, max_value=cfg.maximum_mag)
            save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dir_max_magnitude_{i}_param.png'),
                                f'Maximum Magnitude {measurement_names[-2]}', dvf.magnitude(), res, cfg.maximum_mag)

            dvf = residual_deformation(ref_vol.cpu(), vol_hat_relevant.cpu(), res, ref_msk=ref_msk_np, prefilter=args.prefilter, mode='cubic' if True==args.prefilter else 'bilinear').squeeze() #dvf = residual_deformation(ref_vol.cpu(), vol_hat_relevant.cpu(), res, ref_msk=ref_msk)
            dir_stats[i, -1] = dvf_statistics(dvf, ref_msk, max_value=cfg.maximum_mag)

            save_mag_max_intensity(os.path.join(dst_path, stats_path, f'dir_max_magnitude_{i}_param_{relevant_cnt}.png'),
                                f'Maximum Magnitude {measurement_names[-1]}', dvf.magnitude(), res, cfg.maximum_mag)

        save_ortho_views(f'{i}: Image', ref_vol, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'src_{i}.png', views=cfg.views)

        save_ortho_views(f'{i}: Model', pca_vol, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'model_{i}.png', views=cfg.views)

        save_ortho_views(f'{i}: Residual (Image-Model)', ref_vol - pca_vol, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'diff_{i}.png', views=cfg.views)

        save_ortho_views(f'{i}: Param Model', vol_hat, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'param_model_{i}.png', views=cfg.views)

        save_ortho_views(f'{i}: Residual (Image-Param)', ref_vol - vol_hat, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'param_diff_{i}.png', views=cfg.views)

        save_ortho_views(f'{i}: Residual (Model-Param)', pca_vol - vol_hat, res, pos,
                        dst_path=os.path.join(dst_path, 'pca_eval'), fn=f'model_param_diff_{i}.png', views=cfg.views)

    save_all_mse_plots(pca_mse_values, measurement_names[1:], dst_path, 'full', 'Residual MSE', 4550)
    save_all_mse_plots(pca_msk_mse_values, measurement_names[1:], dst_path, 'msk', 'Masked Residual MSE', 304020)

    dvf_stats = transform_stats(dvf_stats)
    save_all_roc_curves(dvf_stats, measurement_names, os.path.join(dst_path, stats_path),
                        'Residual DVF Magnitude ROC', 'dvf', maximum_mag=cfg.maximum_mag)

    save_cum_roc_curves(dvf_stats, measurement_names, os.path.join(dst_path, stats_path),
                        'Residual DVF Magnitude ROC', 'total_dvf', maximum_mag=cfg.maximum_mag)

    plot_stats(dvf_stats['mag_max'],
            'Residual DVF - Magnitude Maximum', measurement_names[1:],
            fn=os.path.join(dst_path, stats_path, 'dvf_mag_max.png'))

    plot_stats(dvf_stats['mag_mean'],
            'Residual DVF - Magnitude Mean', measurement_names[1:],
            fn=os.path.join(dst_path, stats_path, 'dvf_mag_mean.png'))

    plot_stats(dvf_stats['mag_std'],
            'Residual DVF - Magnitude STD', measurement_names[1:],
            fn=os.path.join(dst_path, stats_path, 'dvf_mag_std.png'))


    if comp_dir:
        dir_stats = transform_stats(dir_stats)
        save_all_roc_curves(dir_stats, measurement_names, os.path.join(dst_path, stats_path),
                            'Residual DIR Magnitude ROC', 'dir', maximum_mag=cfg.maximum_mag)
        save_cum_roc_curves(dir_stats, measurement_names, os.path.join(dst_path, stats_path),
                            'Residual DIR Magnitude ROC', 'total_dir', maximum_mag=cfg.maximum_mag)

        plot_stats(dir_stats['mag_max'],
                'Deformable Image Registration - Magnitude Maximum', measurement_names[1:],
                fn=os.path.join(dst_path, stats_path, 'dir_mag_max.png'))

        plot_stats(dir_stats['mag_mean'],
                'Deformable Image Registration - Magnitude Mean', measurement_names[1:],
                fn=os.path.join(dst_path, stats_path, 'dir_mag_mean.png'))

        plot_stats(dir_stats['mag_std'],
                'Deformable Image Registration - Magnitude STD', measurement_names[1:],
                fn=os.path.join(dst_path, stats_path, 'dir_mag_std.png'))

    with open(os.path.join(dst_path, 'pca_mse_values.csv'), 'w', newline='') as csvfile:
        wrt = csv.writer(csvfile, delimiter=',')
        wrt.writerow([f'{j} Components' for j in range(pca.component_count)] +
                    [f'Parametrization {"amplitude" if args.amplitude_gated else "phase"} gated',
                    f'Reduced ({relevant_cnt}) parametrization {"amplitude" if args.amplitude_gated else "phase"} gated',
                    'Masked'])

        for pmse in pca_mse_values:
            wrt.writerow(pmse.tolist() + ['False'])
        for pmse in pca_msk_mse_values:
            wrt.writerow(pmse.tolist() + ['True'])

    em_cnt = len(cfg.edge_measurements)
    if em_cnt > 0:
        measurement_list = np.array(measurement_list)
        measurement_list = measurement_list.reshape((measurement_list.shape[0], -1, em_cnt))

        for i in range(em_cnt):
            # ['Reference', 'Mean', '1 pc', '2 pc', '3 pc', '4 pc', '5 pc', '6 pc', '7 pc', 'Parameterized', 'Relevant']
            save_measurement_plot(os.path.join(dst_path, f'profile_all_{i}.png'),
                                measurement_names, measurement_list[..., i], args.amplitude_gated)

            col_indices = [0, 2, 3, -4, -2, -1]
            save_measurement_plot(os.path.join(dst_path, f'profile_cmp_{i}.png'),
                                measurement_names, measurement_list[..., i], args.amplitude_gated, col_indices=col_indices)

            col_indices = [0, 2, 3, -4]
            save_measurement_plot(os.path.join(dst_path, f'profile_pca_{i}.png'),
                                measurement_names, measurement_list[..., i], args.amplitude_gated, col_indices=col_indices)

            col_indices = [0, -2, -1]
            save_measurement_plot(os.path.join(dst_path, f'profile_param_{i}.png'),
                                measurement_names, measurement_list[..., i], args.amplitude_gated, col_indices=col_indices)

        with open(os.path.join(dst_path, 'measurements.csv'), 'w', newline='') as csvfile:
            wrt = csv.writer(csvfile, delimiter=',')
            for i in range(em_cnt):
                wrt.writerow(measurement_names)
                for m in measurement_list[..., i]:
                    wrt.writerow(m)

# %%
#Memory clean up
del vols
del pca
del dvfs
import gc
gc.collect()
torch.cuda.empty_cache()
print(f'<<<<<<<<<< Finished reconstruction vols and  generation for {behaviourPrefixedConfigKey}_{patientMRN} >>>>>>>>>>>>>>>')


