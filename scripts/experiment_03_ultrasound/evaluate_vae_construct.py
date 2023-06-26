from tqdm import trange
from PIL import Image
from pathlib import Path
import torch
from torchvision import utils
import math
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.data.datasets import BreastCancerDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as tF
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import (
    multiscale_structural_similarity_index_measure as mmssim,
)


def rgb2gray(img):
    # img [B, C, H, W]
    return ((0.3 * img[:, 0]) + (0.59 * img[:, 1]) + (0.11 * img[:, 2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]


def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b - b.min()) / (b.max() - b.min()) for b in img])


def concat_pair_images(source_img, target_img, width, height, save_path):
    concat_image = Image.new("RGB", (width * 2, height))
    concat_image.paste(source_img, (0, 0))
    concat_image.paste(target_img, (256, 0))
    concat_image.save(save_path)


def evaluate(save_img=False):
    path_out = Path.cwd() / "results/breastCancer/samples_vae_00"
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda")

    dataset = BreastCancerDataset(  #  256x256
        image_resize=(256, 256),
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root="/mnt/d",
    )

    model = VAE(
        in_channels=3,
        out_channels=3,
        emb_channels=8,
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention="none",
        loss=torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6,
    )

    model.load_pretrained(
        Path.cwd() / "runs/experiment03_vae_500_1000ep/last.ckpt",
        strict=True,
    )
    model.eval()
    sample_nums = 10
    index_list = [0,1,2,3,4,254,255,256,257,258]

    # 出图
    results = []
    # for i in range(sample_nums):
    for i in index_list:
        with torch.no_grad():
            data = dataset[i]["source"]
            input = data.unsqueeze(0)
            out, _, emb_loss = model.forward(input)
            # out = (out + 1) / 2
            # out = out.clamp(0, 1)
            results.append(out[0])

    calc_lpips = LPIPS().to(device)
    mmssim_list, mse_list = [], []
    # for i in range(len(results)):
    for i in index_list:
        result = normalize(results[index_list.index(i)])
        source = normalize(dataset[i]["source"])
        target = dataset[i]['target']
        target_str = "malignant" if target else "benigh"
        # source = (source + 1) / 2
        # source = source.clamp(0, 1)
        nrow = int(math.sqrt(result.shape[0]))

        # print(f"source: max={torch.max(source)}, min={torch.min(source)}, avg={torch.mean(source)}")
        # print(f"result: max={torch.max(result)}, min={torch.min(result)}, avg={torch.mean(result)}")

        # evaluation
        if save_img:
            result_path = path_out / f"test_vae_{i}.png"
            source_path = path_out / f"test_vae_source_{i}.png"
            save_path = path_out / f"test_vae_paired_{i}_{target_str}.png"

            utils.save_image(
                result,
                result_path,
                nrow=nrow,
                normalize=True,
                scale_each=True,
            )  # For 2D images: [B, C, H, W]
            utils.save_image(
                dataset[i]["source"],
                source_path,
                nrow=nrow,
                normalize=True,
                scale_each=True,
            )  # For 2D images: [B, C, H, W]
            concat_pair_images(
                Image.open(source_path),
                Image.open(result_path),
                256,
                256,
                save_path,
            )

        source_extend = source.unsqueeze(0)
        result_extend = result.unsqueeze(0)

        calc_lpips.update(source_extend.to(device), result_extend.to(device))
        mmssim_list.append(mmssim(source_extend, result_extend, normalize="relu"))
        mse_list.append(torch.mean(torch.square(source - result)))

    mmssim_list = torch.stack(mmssim_list)
    mse_list = torch.stack(mse_list)
    lpips = 1 - calc_lpips.compute()

    print(f"LPIPS Score: {lpips}")
    print(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
    print(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")


if __name__ == "__main__":
    evaluate(save_img=True)
    # dataset = BreastCancerDataset(  #  256x256
    #     image_resize=(256, 256),
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False,
    #     path_root="/mnt/d",
    # )
    # m_list = []
    # b_list = []
    # for i in range(len(dataset)):
    #     if dataset[i]['target']:
    #         m_list.append(i)
    #     else:
    #         b_list.append(i)
    # print(m_list[:10])
    # print(b_list[:10])
