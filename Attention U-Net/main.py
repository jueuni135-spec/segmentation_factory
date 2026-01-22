from utils import get_data_loaders
from model_v3 import AttentionUNet_MobileNetV3
from train import train_and_test
# from loss import HybridLoss
from loss import FocalTverskyLoss
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = '/content/drive/MyDrive/LikeLion_CV/b. Segmentation_Team/results_exp08_09'
    
    dataloaders = get_data_loaders(batch_size=32)
    
    model = AttentionUNet_MobileNetV3(output_ch=4, pretrained=True).to(device)
    
    # 가중치 설정 및 하이브리드 로스 선언
    final_weights = torch.tensor([1.0, 40.0, 70.0, 2.0]).to(device)
    criterion = FocalTverskyLoss(weights=final_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')

    # 학습 시작
    train_and_test(
        model, dataloaders, optimizer, criterion, 
        num_epochs=30, scheduler=scheduler, scaler=scaler, save_dir=SAVE_DIR
    )

if __name__ == '__main__':
    train()