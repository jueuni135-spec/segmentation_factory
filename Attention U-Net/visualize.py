import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# SAVE_DIR은 이제 함수 내부 인자로 받으므로 전역 변수는 삭제하거나 유지해도 됨
def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# save_dir 인자 추가
def plot_predictions(model, dataloader, device, save_dir, num_samples=4):
    model.eval()
    samples = next(iter(dataloader))
    images, masks = samples['image'], samples['mask']
    
    inputs = images.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    plt.figure(figsize=(18, num_samples * 5))
    color_map = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]])

    for i in range(min(num_samples, len(images))):
        img_show = denormalize(images[i].numpy())
        img_show_255 = (img_show * 255).astype(np.uint8)
        gt_color = color_map[masks[i].numpy()].astype(np.uint8)
        pred_color = color_map[preds[i]].astype(np.uint8)
        
        alpha = 0.5
        overlay_gt = cv2.addWeighted(img_show_255, 1 - alpha, gt_color, alpha, 0)
        overlay_pred = cv2.addWeighted(img_show_255, 1 - alpha, pred_color, alpha, 0)
        
        display_list = [img_show, overlay_gt, overlay_pred]
        title = ['Original', 'GT', 'Pred']
        for j in range(3):
            plt.subplot(num_samples, 3, i*3 + j + 1)
            plt.title(title[j]); plt.imshow(display_list[j]); plt.axis('off')
    
    plt.tight_layout()
    # 경로 수정
    plt.savefig(os.path.join(save_dir, 'overlay_prediction_result.png'))
    plt.close() # 메모리 관리를 위해 창 닫기

# save_dir 인자 추가
def plot_learning_curves(train_losses, val_losses, train_mious, val_mious, save_dir):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Loss Trend'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_mious, label='Train mIoU', color='green')
    plt.plot(val_mious, label='Val mIoU', color='orange')
    plt.title('mIoU Trend'); plt.legend()

    plt.tight_layout()
    # 경로 수정
    plt.savefig(os.path.join(save_dir, 'learning_curves_combined.png'))
    plt.close()