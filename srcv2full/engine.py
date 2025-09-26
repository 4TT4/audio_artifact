import torch
from torchinfo import summary
from tqdm import tqdm
from datetime import datetime
import os

from srcv2full.model import YAMNet
import srcv2full.params as params
from srcv2full.feature_extraction import WaveformToMelSpec

def trivial_collate_fn(inputs):
    return inputs

class YAMNetEngine(torch.nn.Module):
    def __init__(self, model: YAMNet, tt_chunk_size, logger):
        super().__init__()

        self.model = model
        self.tt_chunk_size = tt_chunk_size
        self.logger = logger
    
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.device = torch.device("cuda")
            logger.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        elif torch.backends.mps.is_available():
            logger.info("Using Metal")
            self.device = torch.device("mps")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")
        self.to(self.device)
        self.model.to(self.device)
        
        self.waveform_transform = WaveformToMelSpec(self.device)
        
    def forward(self, inputs):
        overall_preds = []
        for payload in inputs:
            data, sr = payload["data"], payload["sr"]
            data = data.to(self.device)
            sr = torch.tensor(sr, device=self.device)
            mel_spectro, _ = self.waveform_transform(data, sr)
            chunks = mel_spectro.split(self.tt_chunk_size, dim=0)
            accuracies = []
            for chunk in chunks:
                mask = (chunk != 0).any(dim=(1, 2, 3))
                chunk = chunk[mask]
                
                if len(chunk) > 0:  # Only process non-empty chunks
                    pred = self.model(chunk)
                    accuracies.append(pred)
            overall_preds.append(accuracies)
        return overall_preds
    
    def train_yamnet(self, train_loader, val_loader, checkpoint_path, num_labels, num_epochs):
        start = datetime.now()
        self.logger.info("Started training")
        
        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=params.LEARNING_RATE)
        optimizer = torch.optim.Adam([
            {"params": [p for n, p in self.model.named_parameters() if "layer_12" in n or "layer_13" in n], "lr": 1e-4},
            {"params": self.model.classifier.parameters(), "lr": 5e-4}
        ])
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = len(train_loader.dataset)
            # total = len(train_loader.sampler.indices)
            
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
            
            for batch in train_iterator:
                try:
                    outputs = self.forward(batch)[0]
                    if len(outputs) == 0:
                        continue
                        
                    outputs_tensor = outputs[0]  # First chunk's output
                    preds = torch.mean(outputs_tensor, dim=0)
                    actual_label = batch[0]["label"]
                    
                    # FIXED: Use consistent loss calculation
                    loss = criterion(preds.unsqueeze(0), torch.tensor([actual_label], device=self.device))
                    
                    _, predicted = torch.max(preds, dim=0)
                    # FIXED: Proper accuracy calculation
                    correct += int(predicted.item() == actual_label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # FIXED: Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_iterator.set_postfix(loss=loss.item())
                    
                except Exception as e:
                    self.logger.error(f"Error in training batch: {e}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_acc = correct / total if total > 0 else 0
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validation phase
            self.model.eval()
            
            val_loss = 0.0
            correct = 0
            total = len(train_loader.dataset)
            # total = len(val_loader.sampler.indices)
            
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_iterator:
                    try:
                        outputs = self.forward(batch)[0]
                        if len(outputs) == 0:
                            continue
                            
                        outputs_tensor = outputs[0]  # First chunk's output
                        preds = torch.mean(outputs_tensor, dim=0)
                        actual_label = batch[0]["label"]
                        
                        # FIXED: Use consistent loss calculation
                        loss = criterion(preds.unsqueeze(0), torch.tensor([actual_label], device=self.device))
                        
                        _, predicted = torch.max(preds, dim=0)
                        # FIXED: Proper accuracy calculation
                        correct += int(predicted.item() == actual_label)
                        
                        val_loss += loss.item()
                        val_iterator.set_postfix(loss=loss.item())
                        
                    except Exception as e:
                        self.logger.error(f"Error in validation batch: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_acc = correct / total if total > 0 else 0
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"New best model saved with val_acc: {val_acc:.4f}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        end = datetime.now()
        duration = end - start
        self.logger.info(f"Training completed. Runtime: {duration}")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")