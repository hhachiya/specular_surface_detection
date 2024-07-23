import os
import math
import statistics
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import cv2

from config import parse_args, mold_dataset, spherical_mirror_dataset
from metrics import get_maxFscore_and_threshold

def main():
    opt = parse_args()

    """ dataset """
    if "mold" in opt.dataset_path:
        result_dir = "result/plastic_mold"
        _, _, test_loader = mold_dataset(opt)
        result_root = os.path.join(result_dir, opt.write_dir)
        print("Plastic mold dataset")
    elif "spherical" in opt.dataset_path:
        result_dir = "result/spherical_mirror"
        _, _, test_loader = spherical_mirror_dataset(opt)
        result_root = os.path.join(result_dir, opt.write_dir)
        print("Spherical mirror dataset")
    else:
        print("Not found dataset.")
        return
    
    print(f"Result root: {result_root}")
        
    """ model definition """
    from model.network import Network, Encoder
    encoder = Encoder().cuda()
    model = Network().cuda()
    model.eval()
    model_path = os.path.join(result_root, "checkpoint", "refine.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")
    
    '''directory'''
    test_predict = os.path.join(result_root, "test_predict")
    raw_predict = os.path.join(test_predict, "raw_predict")
    thres_predict = os.path.join(test_predict, "thres_predict")
    analysis = os.path.join(test_predict, "analysis")
    os.makedirs(test_predict, exist_ok=True)
    os.makedirs(raw_predict, exist_ok=True)
    os.makedirs(thres_predict, exist_ok=True)
    os.makedirs(analysis, exist_ok=True)
    print("Directories created.")
    
    '''test'''
    max_Fbeta, MAE = [], []
    with torch.no_grad():
        for i, (image, sv, mask, edge, meta) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.cuda()
            sv = sv.cuda()
            mask = mask.cuda()
            edge = edge.cuda()
            
            # predict
            feat_maps = encoder(image)
            preds = model(feat_maps, sv)
            
            # meta read
            origin_RGB = Image.open(meta['image_path'][0])
            origin_mask = Image.open(meta['mask_path'][0])
            origin_w, origin_h = origin_RGB.size
            filename = meta['filename'][0]
            
            # Log meta information
            print(f"Processing {filename} with original size ({origin_h}, {origin_w})")
            
            # post process
            preds_post_process = []
            for j in range(len(preds)):
                pred = torch.sigmoid(preds[j]).cpu().squeeze().numpy()
                pred_resize = cv2.resize(pred, (origin_w, origin_h), interpolation=cv2.INTER_LINEAR)
                preds_post_process.append(pred_resize)
                
                if j == len(preds) - 1:
                    raw_pred_path = os.path.join(raw_predict, filename)
                    Image.fromarray((pred_resize * 255).astype(np.uint8)).save(raw_pred_path)
                    print(f"Saved raw prediction at {raw_pred_path}")

                    max_fbeta, thres = get_maxFscore_and_threshold(mask.cpu().numpy().flatten(), pred.flatten())
                    max_Fbeta.append(max_fbeta)
                    MAE.append(mean_absolute_error(mask.cpu().numpy().flatten(), pred.flatten()))
                    print(f"max Fbeta: {max_Fbeta[-1]:.3f}, MAE: {MAE[-1]:.3f}")
                    thres_predict_map = pred_resize > thres
                    thres_pred_path = os.path.join(thres_predict, filename)
                    Image.fromarray((thres_predict_map * 255).astype(np.uint8)).save(thres_pred_path)
                    print(f"Saved thresholded prediction at {thres_pred_path}")
            
            # analysis visualization 
            col = 4
            num_images = len(preds_post_process) + 2
            row = math.ceil(num_images / col)
            font_size = 36
            plt.figure(figsize=(20, 30))
            
            plt.subplot(row, col, 1)
            plt.imshow(origin_RGB)
            plt.axis('off')
            plt.title("image", fontsize=font_size)
            plt.subplot(row, col, 2)
            plt.imshow(origin_mask, cmap='gray')
            plt.axis('off')
            plt.title("mask", fontsize=font_size)
            
            for k in range(len(preds_post_process)):
                plt.subplot(row, col, k + 3)
                plt.imshow(preds_post_process[k], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                if k < 4:
                    plt.title(f"RCCL: {4 - k}", fontsize=font_size)
                elif k < 8:
                    plt.title(f"SSF: {8 - k}", fontsize=font_size)
                elif k < 12:
                    plt.title(f"SH: {12 - k}", fontsize=font_size)
                elif k == 12:
                    plt.title(f"boundary", fontsize=font_size)
                elif k == 13:
                    plt.title(f"final", fontsize=font_size)
                else:
                    pass
                    
            plt.tight_layout()
            analysis_path = os.path.join(analysis, filename)
            plt.savefig(analysis_path)
            plt.close()
            print(f"Saved analysis visualization at {analysis_path}")
    
    avg_max_Fbeta = statistics.mean(max_Fbeta)
    avg_MAE = statistics.mean(MAE)
    print(f"avg max Fbeta: {avg_max_Fbeta}, avg MAE: {avg_MAE}")
    std_max_Fbeta = statistics.stdev(max_Fbeta)
    std_MAE = statistics.stdev(MAE)
    print(f"std max Fbeta: {std_max_Fbeta}, std MAE: {std_MAE}")
    
    with open(os.path.join(test_predict, "result.txt"), "w") as f:
        f.write(f"avg max Fbeta: {avg_max_Fbeta}, avg MAE: {avg_MAE}\n")
        f.write(f"std max Fbeta: {std_max_Fbeta}, std MAE: {std_MAE}\n")
    print("Results saved to result.txt")
    print("finish")

if __name__ == "__main__":
    main()