import random
import pdb
import os
import math

import torch
from torch import optim
import numpy as np
from torchmetrics.classification import BinaryFBetaScore
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from config import parse_args, mold_dataset, spherical_mirror_dataset
import earlystop


def plot_loss_gragh(t_loss, v_loss, t_score, v_score, save_path):
    fig = plt.figure(figsize=(25,9))
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Loss", fontsize=18)
    ax1.set_xlabel("Epoch",fontsize=18)
    ax1.set_ylabel("Loss",fontsize=18)
    ax1.plot(t_loss, label="train", marker='o')
    ax1.plot(v_loss, label="valid", marker='o')
    ax1.tick_params(axis='both',labelsize=15)
    ax1.grid()
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("BinaryFbetaScore", fontsize=18)
    ax2.set_xlabel("Epoch", fontsize=18)
    ax2.set_ylabel("IoU", fontsize=18)
    ax2.plot(t_score, label="train", marker='o')
    ax2.plot(v_score, label="valid", marker='o')
    ax2.tick_params(axis='both',labelsize=15)
    ax2.grid()
    ax2.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    opt = parse_args()

    """ dataset """
    if "mold" in opt.dataset_path:
        result_dir = "result/plastic_mold"
        train_loader, val_loader, _ = mold_dataset(opt)
        result_root = os.path.join(result_dir, opt.write_dir)
        print("Plastic mold dataset")
    elif "spherical" in opt.dataset_path:
        result_dir = "result/spherical_mirror"
        train_loader, val_loader, _ = spherical_mirror_dataset(opt)
        result_root = os.path.join("result/spherical_mirror", opt.write_dir)
        print("Spherical mirror dataset")
    else:
        print("Not found dataset.")

    """ directory structure """
    result_root_check = os.path.join(result_root, "checkpoint")
    result_root_output = os.path.join(result_root, "output")
    result_root_check_output = os.path.join(result_root, "validation_predict")
    result_root_plot = os.path.join(result_root, "loss_metrics_plot")
    component_param_path = os.path.join(result_root_check, f"{opt.mode}.pth")
    
    # Make directories
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(result_root_check, exist_ok=True)
    os.makedirs(result_root_output, exist_ok=True)
    os.makedirs(result_root_check_output, exist_ok=True)
    os.makedirs(result_root_plot, exist_ok=True)
    
    # network setting
    from model.network import Encoder
    encoder = Encoder().cuda() # freezed
    
    if opt.mode == "rccl":
        from model.network import RCCL_SubNet
        model = RCCL_SubNet().cuda()
    
    elif opt.mode == "ssf":
        from model.network import SSF_SubNet
        model = SSF_SubNet().cuda()
        
    elif opt.mode == "sh":
        from model.network import SH_SubNet
        model = SH_SubNet().cuda()

    elif opt.mode == "refine":
        from model.network import Network
        model = Network().cuda()
        rccl_param_path = os.path.join(result_root_check, "rccl.pth") 
        ssf_param_path = os.path.join(result_root_check, "ssf.pth") 
        sh_param_path = os.path.join(result_root_check, "sh.pth") 
        
        model.rccl_net.load_state_dict(torch.load(rccl_param_path))
        model.ssf_net.load_state_dict(torch.load(ssf_param_path))
        model.sh_net.load_state_dict(torch.load(sh_param_path))
        
    else:
        raise ValueError("Not found mode.")
    
    ''' setting '''
    optimizer = optim.Adam(model.parameters())
    metrics_fn = BinaryFBetaScore(beta=0.5)
    es = earlystop.EarlyStopping(
                                verbose=True,
                                patience=opt.patient, 
                                path=component_param_path
                                )
    if opt.mode == "refine":
        from loss.dice_bce import LossRefine
        loss = LossRefine(W_map=5, W_final=1)
    else:
        from loss.dice_bce import LossComponent
        loss = LossComponent(W_map=1, W_final=2)

    ''' learning loop '''
    train_loss = []
    train_metrics = []
    val_loss = []
    val_metrics = []
    
    for epoch in range(opt.epochs):
        print('\nEpoch: {}'.format(epoch + 1))
        
        # train
        train_epoch_loss, train_epoch_metrics = 0, 0
        model.train()
        for image, sv, gt_mask, gt_edge, meta in tqdm(train_loader):
            image, sv, gt_mask, gt_edge = image.cuda(), sv.cuda(), gt_mask.cuda(), gt_edge.cuda()
            optimizer.zero_grad()
            # model prediction
            feat_maps = encoder(image)
            pred = model(feat_maps, sv)
            
            if opt.mode != "refine":
                pred = pred[:-1]
            
            # loss calculation
            loss_output = loss(pred, gt_mask, gt_edge)
            loss_output.backward()
            optimizer.step()
            
            train_epoch_loss += loss_output.item()
            train_epoch_metrics += metrics_fn(torch.sigmoid(pred[-1]).cpu(), gt_mask.cpu()).item()
            
        train_loss.append(train_epoch_loss / len(train_loader))
        train_metrics.append(train_epoch_metrics / len(train_loader))
        print(f"train loss: {train_epoch_loss / len(train_loader)}, train metrics: {train_epoch_metrics / len(train_loader)}")
        
        # validation
        val_epoch_loss, val_epoch_metrics = 0, 0
        model.eval()
        with torch.no_grad():
            for image, sv, gt_mask, gt_edge, meta in tqdm(val_loader):
                image, sv, gt_mask, gt_edge = image.cuda(), sv.cuda(), gt_mask.cuda(), gt_edge.cuda()
                
                feat_maps = encoder(image)
                pred = model(feat_maps, sv)
                
                if opt.mode != "refine":
                    pred = pred[:-1]
            
                loss_output = loss(pred, gt_mask, gt_edge)
                val_epoch_loss += loss_output.item()
                val_epoch_metrics += metrics_fn(torch.sigmoid(pred[-1]).cpu(), gt_mask.cpu()).item()
        
        val_avgloss = val_epoch_loss / len(val_loader)
        val_avgmetrics = val_epoch_metrics / len(val_loader)
        val_loss.append(val_avgloss)
        val_metrics.append(val_avgmetrics)
        print(f"val loss: {val_avgloss}, val metrics: {val_avgmetrics}")
        
        # validation output
        num_graghs = len(pred)
        col = 4
        row = math.ceil((num_graghs + 3) / col)
        font_size = 36
        plt.figure(tight_layout=True, figsize=(20, 30))
        
        for i in range(num_graghs):
            pred_arr = torch.sigmoid(pred[i]).cpu().squeeze().numpy()
            plt.subplot(row, col, i + 1, label=f"pred_{i}")
            plt.imshow(pred_arr, cmap='gray', vmin=0, vmax=1)
            if i < 4:
                plt.title(f"RCCL: {4 - i}", fontsize=font_size)
            elif i < 8:
                plt.title(f"SSF: {8 - i}", fontsize=font_size)
            elif i < 12:
                plt.title(f"SH: {12 - i}", fontsize=font_size)
            elif i == 12:
                plt.title(f"boundary", fontsize=font_size)
            elif i == 13:
                plt.title(f"final", fontsize=font_size)
        
        plt.subplot(row, col, num_graghs + 1, label="image")
        
        plt.imshow(Image.open(meta['image_path'][0]).convert("RGB").resize((opt.resize, opt.resize)))
        plt.title("image", fontsize=font_size)
        plt.subplot(row, col, num_graghs + 2, label="mask")
        plt.imshow(gt_mask.cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title("mask", fontsize=font_size)
        plt.subplot(row, col, num_graghs + 3, label="detected_mirror")
        pred_final = torch.sigmoid(pred[-1]).cpu().squeeze().numpy()
        image_np = np.clip(image.cpu().squeeze().permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
        masking_img = (image_np * pred_final[:, :, np.newaxis]).astype(np.uint8)
        plt.imshow(masking_img)
        plt.title("detected mirror", fontsize=font_size)
        plt.axis('off')
        plt.savefig(os.path.join(result_root_check_output, f"{opt.mode}.png"))
        plt.close()

        if epoch % 5 == 0:
            save_loss_gragh_path = os.path.join(result_root_plot, f"{opt.mode}.png")
            print(save_loss_gragh_path)
            plot_loss_gragh(train_loss, val_loss, train_metrics, val_metrics, save_loss_gragh_path)
        
        es(val_avgloss, model)
        if es.early_stop:
            print("Early Stopping.")
            break


if __name__ == '__main__':
    main()