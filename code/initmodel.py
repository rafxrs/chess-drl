import torch
import os
from modelbuilder import RLModelBuilder
import config

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

model = RLModelBuilder(
    config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
).build_model()
torch.save(model.state_dict(), "models/initial_model.pt")

print(f"Initial model created and saved to models/initial_model.pt")