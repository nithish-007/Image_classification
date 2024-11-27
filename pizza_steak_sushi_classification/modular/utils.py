import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

        # create target directory to save the model
        target_dir_path = Path(target_dir)
        if target_dir_path.is_dir():
            print(f"Target dir already exist {target_dir_path} skipping to create a dir")
        else:
            target_dir_path.mkdir(parents = True, exist_ok = True)

        # create model save path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        # save the model save_dict()
        print(f"[Info] Saving model to: {model_save_path}")
        torch.save(obj = model.save_dict(), f = model_save_path)