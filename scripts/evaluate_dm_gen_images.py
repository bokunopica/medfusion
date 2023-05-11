from pathlib import Path
import logging
from datetime import datetime
from tqdm import trange
from PIL import Image
import numpy as np
import torch
from medical_diffusion.data.datasets import (
    CheXpert_2_Dataset_Cardiomegaly,
    CheXpert_2_Dataset_evaluate,
)
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore as IS

from medical_diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall


# ----------------Settings --------------
n_samples = 500
max_samples = None  # set to None for all
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd() / "results" / "CheXpert" / "metrics"
path_out.mkdir(parents=True, exist_ok=True)
gen_image_path = Path.cwd() / "results" / "CheXpert" / "samples_gen"

# -----------------helpers ---------------
pil2torch = (
    lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) / 255.0
)  # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b - b.min()) / (b.max() - b.min()) for b in img])

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out / f"metrics_{current_time}.log", "w"))

# ------------- Init Metrics ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
calc_fid = FID().to(device)  # requires uint8
# calc_is = IS(splits=1).to(device) # requires uint8, features must be 1008 see https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L603
calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)

# ------------- dataset --------------------------

evaluate_ds = CheXpert_2_Dataset_evaluate(  #  256x256
    image_resize=(256, 256),
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root="/home/Slp9280082/"
)


# --------------- Start Calculation -----------------
for i in trange(n_samples):
    real_img = evaluate_ds[i]["source"].type(torch.uint8).unsqueeze(0).to(device)
    cond = evaluate_ds.transfer_target(
        evaluate_ds.labels.iloc[i]['Cardiomegaly']
    )
    fake_img_path = gen_image_path / f"evaluate_{i}.png"
    fake_img = evaluate_ds.transform(Image.open(fake_img_path).convert("RGB")).type(torch.uint8).unsqueeze(0).to(device)
    calc_fid.update(real_img, real=True)
    calc_pr.update(real_img, real=True)
    calc_fid.update(fake_img, real=False)
    calc_pr.update(fake_img, real=False)        


# -------------- Summary -------------------
fid = calc_fid.compute()
logger.info(f"FID Score: {fid}")

# is_mean, is_std = calc_is.compute()
# logger.info(f"IS Score: mean {is_mean} std {is_std}")

precision, recall = calc_pr.compute()
logger.info(f"Precision: {precision}, Recall {recall} ")
