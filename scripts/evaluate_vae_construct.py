from pathlib import Path
import torch
from torchvision import utils
import math
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.data.datasets import CheXpert_2_Dataset_test
from PIL import Image
from torchvision import transforms


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


if __name__ == "__main__":
    path_out = Path.cwd() / "results/CheXpert/samples_vae_00"
    path_out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda")

    dataset = CheXpert_2_Dataset_test(  #  256x256
        image_resize=(256, 256),
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        # path_root = '/home/gustav/Documents/datasets/CheXpert/preprocessed_tianyu'
        # path_root = '/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/preprocessed_tianyu'
        path_root="/mnt/c/MyCodes/",
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
        Path.cwd() / "runs/2023_05_07_170136/epoch=999-step=62000.ckpt", strict=True
    )
    # model.eval()
    sample_nums = 10

    results = []
    for i in range(sample_nums):
        with torch.no_grad():
            data = dataset[i]["source"]
            # the conv layer of model needs dim0=8 &4D Tensor input(why 8?)
            input = data.unsqueeze(0)
            for _ in range(7):
                input = torch.cat((input, data.unsqueeze(0)), dim=0)
            out, _, emb_loss = model.forward(input)
            out = (out + 1) / 2
            out = out.clamp(0, 1)
            results.append(out[0])

    for i in range(len(results)):
        result = results[i]
        result = (result + 1) / 2
        result = result.clamp(0, 1)
        source = dataset[i]["source"]
        nrow = int(math.sqrt(result.shape[0]))

        result_path = path_out / f"test_vae_{i}.png"
        source_path = path_out / f"test_vae_source_{i}.png"
        save_path = path_out / f"test_vae_paired_{i}.png"

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
        
        


        

    # results = (results+1)/2
    # results = results.clamp(0, 1)
    # utils.save_image(results, path_out/f'test_{cond}.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
    # print(results)

    # diff = torch.abs(normalize(rgb2gray(images[1]))-normalize(rgb2gray(images[0]))) # [0,1] -> [0, 1]
    # # diff = torch.abs(images[1]-images[0])
    # utils.save_image(diff, path_out/'diff.png', nrow=int(math.sqrt(results.shape[0])), normalize=True, scale_each=True) # For 2D images: [B, C, H, W]
