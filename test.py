'''
Test Vid4 datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import argparse

import Dataset.data_utils as util
import models.archs.DfRes_seprecon_selfattn_nores_nlsa as DfRes_arch_nores_nlsa

import time
def main(args):
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'evenodd' 
    #### model ###############################################################
               
    model_path = '../experiments/DfRes_seprecon_nores_covoffset_EkSA_dim64k50_trainonYOUKU/models/150000_G.pth'
    N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 7
    model = DfRes_arch_nores_nlsa.DfRes_nores_nlsa(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in, w_TSA=False)
    
    
    test_dataset_folder  = '../datasets/vid/interlaced/'
    GT_dataset_folder  = '../datasets/vid/progressive/'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2
    padding = 'replicate'
    save_imgs = False
    
    save_folder = '../results/{}_DfRes_seprecon_nores_covoffset_EkSA_dim64k50_trainonYOUKU_models_150000_G_testonVid4'.format(data_mode)
    
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_ssim_l = []
    avg_time_l = []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    print(subfolder_l)
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    print(subfolder_GT_l)
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))

        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(util.read_img(None, img_GT_path))

        avg_psnr, avg_psnr_sub = 0, []
        avg_ssim, avg_ssim_sub = 0, []
        avg_time, avg_time_sub = 0, []

        Bufinit_idx = 0
        Bufmax_idx = -1
        max_idx = 0
        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            imname = img_path.split('/')[-1]
            Sqmax_idx = int(imname.split('_')[-1].split('-')[0])
            Sqstart_idx=int(imname.split('-')[-1].split('.')[0])
            if Sqstart_idx == 1:
                Bufinit_idx = Bufmax_idx +1 
                Bufmax_idx = Bufmax_idx + Sqmax_idx
                max_idx = Bufmax_idx
            print(Bufinit_idx, img_idx, max_idx, Sqmax_idx, N_in)
            img_name = osp.splitext(osp.basename(img_path))[0]

            select_idx = util.index_generation_evenodd(Bufinit_idx, img_idx, max_idx, N_in, padding=padding)
            imgs_in = []
            for v in select_idx:
                print('last3v:',v)
                img_LQ_2 = imgs_LQ[v]
                imgs_in.append(img_LQ_2)

            c, h, w = imgs_in[0].shape
            print(imgs_in[0].shape,imgs_in[1].shape,imgs_in[2].shape,imgs_in[3].shape,imgs_in[4].shape)
            if data_mode == 'evenodd' and (img_idx-Bufinit_idx) % 2 == 0:
                imgs_in.append(np.zeros((c,h,w),dtype=int))
            elif data_mode == 'evenodd' and (img_idx-Bufinit_idx) % 2 != 0:
                imgs_in.append(np.ones((c,h,w),dtype=int))
                
            imgs_in = np.stack(imgs_in, axis=0)
            imgs_in = torch.from_numpy(np.ascontiguousarray(imgs_in)).float()
            imgs_in = imgs_in.unsqueeze(0).to(device)

            start_time = time.time()
            
            output = util.single_forward(model, imgs_in)
            
            end_time = time.time()
            total_time = end_time - start_time

            output = util.tensor2img(output.squeeze(0))
            

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx//2])
            crt_psnr = util.calc_psnr(GT,output)
            crt_ssim = util.calc_ssim(GT,output)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB \tSSIM: {:.6f} \tTIME: {:.6f} s'.format(img_idx + 1, img_name, crt_psnr, crt_ssim, total_time))

            avg_psnr_sub.append(crt_psnr)
            avg_ssim_sub.append(crt_ssim)
            avg_time_sub.append(total_time)

        avg_psnr = sum(avg_psnr_sub) / len(avg_psnr_sub)
        avg_ssim = sum(avg_ssim_sub) / len(avg_ssim_sub)
        avg_time = sum(avg_time_sub) / len(avg_time_sub)
        
        avg_psnr_l.append(avg_psnr)
        avg_ssim_l.append(avg_ssim)
        avg_time_l.append(avg_time)

        logger.info('Folder {} - Average PSNR: {:.6f} dB  SSIM: {:.6f}  TIME: {:.6f} s for {} frames.'
                    .format(subfolder_name, avg_psnr, avg_ssim, avg_time, len(avg_psnr_sub)))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, ssim, times in zip(subfolder_name_l, avg_psnr_l, avg_ssim_l, avg_time_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB, SSIM: {:.6f}, TIME: {:.6f} s. '
                    .format(subfolder_name, psnr, ssim, times))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.6f} dB, SSIM: {:.6f}, TIME: {:.6f} s for {} clips. '
                .format(sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_ssim_l) / len(avg_ssim_l), sum(avg_time_l) / len(avg_time_l), len(subfolder_l)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on Vid dataset')
    args = parser.parse_args()
    
    main(args)
