import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dual_stream_net import DualStreamNet
from utils.data_utils import get_dataloader
from utils.loss import total_loss
from config import Config

def train_model(model, dataloader, config):
    optimizer = optim.RAdam(model.parameters(), lr=config.LEARNING_RATE)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for static_input, dynamic_input, target in dataloader:
            static_input = static_input.to(config.DEVICE)
            dynamic_input = dynamic_input.to(config.DEVICE)
            target = target.to(config.DEVICE)
            
            optimizer.zero_grad()
            pred = model(static_input, dynamic_input)
            loss = total_loss(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    config = Config()
    dataloader = get_dataloader(config.IMAGE_DIR, config.IMG_SIZE, config.BATCH_SIZE)
    model = DualStreamNet().to(config.DEVICE)
    train_model(model, dataloader, config)