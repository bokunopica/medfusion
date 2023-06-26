from pathlib import Path
import torch
from torchvision import utils
import math
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.data.datasets import CheXpert_2_Dataset_Evaluate, CheXpert_2_Dataset


def rgb2gray(img):
    # img [B, C, H, W]
    return ((0.3 * img[:, 0]) + (0.59 * img[:, 1]) + (0.11 * img[:, 2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]


def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b - b.min()) / (b.max() - b.min()) for b in img])


if __name__ == "__main__":
    path_out = Path.cwd() / "results/CheXpert/experiment_02/samples_gen_noise"
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1)
    device = torch.device("cuda")

    # ------------ Load Model ------------
    # pipeline = DiffusionPipeline.load_best_checkpoint(path_run_dir)

    pipeline = DiffusionPipeline.load_from_checkpoint(
        "runs/experiment02_dm_20kl_140ep/epoch=140-step=88000.ckpt",
        strict=False
    )
    pipeline.to(device)

    # --------- Generate Samples  -------------------
    steps = 150
    use_ddim = True
    images = {}
    n_samples = 1
    # evaluate_ds = CheXpert_2_Dataset_Evaluate(  #  256x256
    # evaluate_ds = CheXpert_2_Dataset(  #  256x256
    #     image_resize=(256, 256),
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False,
    #     path_root="/mnt/c/data/chexpert"
    # )

    un_cond = None

    for i in range(n_samples):
        img_size = (8, 32, 32)
        # condition = torch.tensor([cond], device=device) if cond is not None else None
        # data = evaluate_ds[i]
        # row = evaluate_ds.labels.iloc[i]
        # condition = CheXpert_2_Dataset.get_prompt_of_target(row)
        condition = "A photo of a lung xray"
        with torch.no_grad():
            result = pipeline.sample_with_denoise_step(
                1,
                img_size,
                condition=condition,
                un_cond=un_cond,
                steps=steps,
                use_ddim=use_ddim,
            )
            print(result.shape)
            print(type(result))

            result = (result + 1) / 2  # Transform from [-1, 1] to [0, 1]
            result = result.clamp(0, 1)
            # # results = normalize(results)
            utils.save_image(
                result,
                path_out / f"evaluate_{i}_{condition}_{steps}.png",
                nrow=int(math.sqrt(result.shape[0])),
                normalize=True,
                scale_each=True,
            )  # For 2D images: [B, C, H, W]