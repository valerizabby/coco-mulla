from coco_mulla.models import CoCoMulla
from config import TrainCfg

device = "cpu"

sample_sec = TrainCfg.sample_sec
latent_dim = 12
num_layers = 48

model = CoCoMulla(sample_sec, num_layers=num_layers, latent_dim=latent_dim)
model.to(device)

model.load_weights("checkpoints/best_model.pth")

model.eval()