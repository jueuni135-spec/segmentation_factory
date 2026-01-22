import copy, time, torch, os, sys
import numpy as np
from tqdm import tqdm
from loss import calculate_miou
from visualize import plot_learning_curves, plot_predictions

def train_and_test(model, dataloaders, optimizer, criterion, num_epochs=30, scheduler=None, scaler=None, save_dir='/content/drive/MyDrive/results'):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'train_log.txt')
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0
    best_class_ious = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, val_losses, train_mious, val_mious = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        epoch_start = f"\n🚀 Epoch {epoch}/{num_epochs}\n" + "="*30 + "\n"
        print(epoch_start)
        with open(log_path, 'a') as f: f.write(epoch_start)

        for phase in ['training', 'validation']:
            model.train() if phase == 'training' else model.eval()
            running_loss, running_miou = 0.0, 0.0
            running_class_ious = np.zeros(4)

            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize():10}")
            for sample in pbar:
                inputs, masks = sample['image'].to(device), sample['mask'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                        outputs = model(inputs)
                        loss = criterion(outputs, masks)

                    if phase == 'training':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                batch_miou, batch_class_ious = calculate_miou(outputs.detach().cpu(), masks.detach().cpu())
                running_loss += loss.item() * inputs.size(0)
                running_miou += batch_miou * inputs.size(0)
                running_class_ious += np.array([0 if np.isnan(x) else x for x in batch_class_ious]) * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_miou = running_miou / len(dataloaders[phase].dataset)
            epoch_class_ious = running_class_ious / len(dataloaders[phase].dataset)

            if phase == 'training':
                train_losses.append(epoch_loss); train_mious.append(epoch_miou)
            else:
                val_losses.append(epoch_loss); val_mious.append(epoch_miou)
                if epoch_miou > best_miou:
                    best_miou = epoch_miou
                    best_class_ious = epoch_class_ious
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                if scheduler: scheduler.step(epoch_loss)

            res_log = f"📊 {phase:10} | Loss: {epoch_loss:.4f} | mIoU: {epoch_miou:.4f}\n" + \
                      f"   └─ IoU [BG: {epoch_class_ious[0]:.4f} | Poll: {epoch_class_ious[1]:.4f} | Dam: {epoch_class_ious[2]:.4f} | Out: {epoch_class_ious[3]:.4f}]\n"
            print(res_log)
            with open(log_path, 'a') as f: f.write(res_log)

        # 매 에폭마다 그래프와 시각화 강제 업데이트 (백그라운드 유실 방지)
        plot_learning_curves(train_losses, val_losses, train_mious, val_mious, save_dir)
        plot_predictions(model, dataloaders['validation'], device, save_dir, num_samples=4)

    print(f"\n🏆 Best Validation mIoU: {best_miou:.4f}")
    print(f"🥇 Best Class IoU -> BG: {best_class_ious[0]:.4f} | Poll: {best_class_ious[1]:.4f} | Dam: {best_class_ious[2]:.4f} | Out: {best_class_ious[3]:.4f}")
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    return model