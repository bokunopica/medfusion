import os
import torch.utils.data as data
import torch
from torch import nn
from pathlib import Path
from torchvision import transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from medical_diffusion.data.augmentation.augmentations_2d import (
    Normalize,
    ToTensor16bit,
)
from medical_diffusion.utils.train_utils import PyObjectCache, MemStorageCache


DISEASE_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers=[],
        crawler_ext="tif",  # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform=None,
        image_resize=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
        use_cache=False
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        self.use_cache=use_cache
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext)

        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(image_resize)
                    if image_resize is not None
                    else nn.Identity(),
                    T.RandomHorizontalFlip()
                    if augment_horizontal_flip
                    else nn.Identity(),
                    T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                    T.CenterCrop(image_crop)
                    if image_crop is not None
                    else nn.Identity(),
                    T.ToTensor(),
                    # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                    # ToTensor16bit(),
                    # Normalize(), # [0, 1.0]
                    # T.ConvertImageDtype(torch.float),
                    T.Normalize(
                        mean=0.5, std=0.5
                    ),  # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        # img = Image.open(path_item)
        img = self.load_item(path_item)
        return {"uid": rel_path_item.stem, "source": self.transform(img)}

    def load_item(self, path_item):
        return Image.open(path_item).convert("RGB")
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [
            path.relative_to(path_root)
            for path in Path(path_root).rglob(f"*.{extension}")
        ]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(
            self.path_root.parent / "train_labels.csv", index_col="challenge_id"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root / f"{uid}.jpg"
        img = self.load_item(path_item)
        str_2_int = {"NRG": 0, "RG": 1}  # RG = 3270, NRG = 98172
        target = str_2_int[self.labels.loc[uid, "class"]]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["class"].value_counts(
            normalize=True
        )  # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]["class"]
            weights[index] = weight_per_class[target]
        return weights

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {"MSIMUT": 0, "MSS": 1}
        target = str_2_int[path_item.parent.name]  #
        return {"uid": uid, "source": self.transform(img), "target": target}


class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {
            "MSIH": 0,
            "nonMSIH": 1,
        }  # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {"source": self.transform(img), "target": target}


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent / f"{mode}.csv", index_col="Path")
        self.labels = labels.loc[labels["Frontal/Lateral"] == "Frontal"].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[
            self.labels["Sex"] == "Unknown", "Sex"
        ] = "Female"  # Affects 1 case, must be "female" to match stats in publication
        self.labels.fillna(2, inplace=True)  # TODO: Find better solution,
        str_2_int = {
            "Sex": {"Male": 0, "Female": 1},
            "Frontal/Lateral": {"Frontal": 0, "Lateral": 1},
            "AP/PA": {"AP": 0, "PA": 1},
        }
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root / rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(
            self.labels.loc[uid, "Cardiomegaly"] + 1, dtype=torch.long
        )  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {"uid": uid, "source": self.transform(img), "target": target}

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []


# class CheXpert_2_Dataset(SimpleDataset2D):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         labels = pd.read_csv(
#             self.path_root / "labels/cheXPert_label.csv",
#             index_col=["Path", "Image Index"],
#         )  # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
#         labels = labels.loc[labels["fold"] == "train"].copy()
#         labels = labels.drop(labels="fold", axis=1)

#         labels2 = pd.read_csv(self.path_root / "labels/train.csv", index_col="Path")
#         labels2 = labels2.loc[labels2["Frontal/Lateral"] == "Frontal"].copy()
#         labels2 = labels2[
#             [
#                 "Cardiomegaly",
#             ]
#         ].copy()
#         labels2[
#             (labels2 < 0) | labels2.isna()
#         ] = 2  # 0 = Negative, 1 = Positive, 2 = Uncertain
#         labels = labels.join(
#             labels2["Cardiomegaly"],
#             on=[
#                 "Path",
#             ],
#             rsuffix="_true",
#         )
#         # labels = labels[labels['Cardiomegaly_true']!=2]

#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         path_index, image_index = self.labels.index[index]
#         path_item = self.path_root / "train" / f"{image_index:06}.png"
#         img = self.load_item(path_item)
#         uid = image_index
#         target = int(self.labels.loc[(path_index, image_index), "Cardiomegaly"])
#         # return {'uid':uid, 'source': self.transform(img), 'target':target}
#         return {"source": self.transform(img), "target": target}

#     @classmethod
#     def run_item_crawler(cls, path_root, extension, **kwargs):
#         """Overwrite to speed up as paths are determined by .csv file anyway"""
#         return []

#     def get_weights(self):
#         n_samples = len(self)
#         weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
#         # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
#         weights = [0] * n_samples
#         for index in range(n_samples):
#             target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
#             weights[index] = weight_per_class[target]
#         return weights


class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        self.count = kwargs.pop("count", 1000)
        self.data_folder_name = kwargs.pop(
            "data_folder_name", "CheXpert-v1.0-Resample@256x256"
        )
        self.embedder_type = kwargs.pop(
            "embedder_type", 0
        )  # 0-LableEmbedder 1-RapBertEmbedder
        super().__init__(*args, **kwargs)
        # labels = pd.read_csv(self.path_root / "CheXpert-v1.0" / "train.csv")
        labels = pd.read_csv(self.path_root / self.data_folder_name / "train.csv")
        labels = labels[labels["Frontal/Lateral"] == "Frontal"]
        labels = labels.iloc[: self.count]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        image_path = row["Path"]
        image_path = f"{self.data_folder_name}/{'/'.join(image_path.split('/')[1:])}"
        # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        raw_target = row["Cardiomegaly"]
        if self.embedder_type == 0:
            target = self.transfer_label(raw_target)
        elif self.embedder_type == 1:
            target = self.get_prompt_of_target(row)
        else:
            target = None
        if self.use_cache:
            source = self.load_source_cache(image_path)
        else:
            source = self.load_source(image_path)

        result = {"source": source, "target": target}
        return result

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
            weights[index] = weight_per_class[target]
        return weights

    def load_item(self, path_item):
        return Image.open(path_item)

    def load_source(self, image_path):
        path_item = self.path_root / image_path
        img = self.load_item(path_item)
        img = img.convert("RGB")
        source = self.transform(img)
        return source

    def load_source_cache(self, image_path):
        # cache = PyObjectCache()
        cache = MemStorageCache()
        image = cache.get(image_path)
        path_item = self.path_root / image_path
        if image is None:
            image = self.load_item(path_item)
            cache.set(image_path, image)
            image = image.convert("RGB")
        source = self.transform(image)
        return source

    @classmethod
    def transfer_label(cls, raw_target):
        if raw_target is np.nan:
            target = 0
        elif raw_target == 1.0:
            target = 1
        else:
            target = 0
        return target

    # @classmethod
    # def transfer_label_3_categories(cls, raw_target):
    #     """
    #     * 1.0 - The label was positively mentioned in the associated study, and is present in one or more of the corresponding images
    #         * e.g. "A large pleural effusion"
    #     * 0.0 - The label was negatively mentioned in the associated study, and therefore should not be present in any of the corresponding images
    #         * e.g. "No pneumothorax."
    #     * -1.0 - The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
    #         * Explicit uncertainty: "The cardiac size cannot be evaluated."
    #         * Ambiguous language: "The cardiac contours are stable."
    #     * Missing (empty element) - No mention of the label was made in the report

    #     """
    #     if raw_target is np.nan:
    #         target = 0
    #     elif raw_target == 1.0:
    #         target = 1
    #     else:
    #         target = 0
    #     return target
    @classmethod
    def get_prompt_of_target(cls, row):
        """
        input
        """
        positive_disease_list = []
        prompt = "A photo of a lung xray"
        for disease in DISEASE_COLUMNS:
            if cls.transfer_label(row[disease]):
                positive_disease_list.append(disease)

        if positive_disease_list:
            if len(positive_disease_list) == 1:
                positive_str = positive_disease_list[0]
            else:
                positive_str = ",".join(positive_disease_list[:-1])
                positive_str += f" and {positive_disease_list[-1]}"
            prompt += f" with {positive_str}"

        return prompt


class CheXpert_2_Dataset_Evaluate(CheXpert_2_Dataset):
    def __init__(self, *args, **kwargs):
        self.start = kwargs.pop("start", 20000)
        self.count = kwargs.pop("count", 500)
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(self.path_root / self.data_folder_name / "train.csv")
        labels = labels[labels["Frontal/Lateral"] == "Frontal"]
        labels = labels.iloc[self.start : (self.start + self.count)]
        self.labels = labels


class CheXpert_2_Dataset_Cardiomegaly(CheXpert_2_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = self.labels[self.labels["Cardiomegaly"] == 1]


class CheXpert_2_Dataset_Cardiomegaly(CheXpert_2_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = self.labels[self.labels["Cardiomegaly"] != 1]


class BreastCancerDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # labels = pd.read_csv(self.path_root / "CheXpert-v1.0" / "train.csv")
        self._init_label()

    def _init_label(self):
        self.labels = list()
        idx = 0
        for path_obj in os.walk(self.path_root / "ultrasound breast classification"):
            # path_obj[0]: cur path ;[1]: child folder; [2]: child file
            for filename in path_obj[2]:
                if self.file_name_check(filename):
                    self.labels.append(
                        (
                            idx,
                            f"{path_obj[0]}/{filename}",
                            self.benign_or_malignant(filename),
                        )
                    )
                    idx += 1

    def file_name_check(self, filename: str):
        found = False
        for char in filename:
            if char == "-":
                found = True
                break
        return not found

    @classmethod
    def benign_or_malignant(cls, filename):
        """
        benign or unknown: 0
        malignant: 1
        """
        if "malignant" in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels[index] # [0] idx [1] path [2] severty label 1=maligant 0=benign 
        image_path = row[1]
        target = row[2]
        if self.use_cache:
            source = self.load_source_cache(image_path)
        else:
            source = self.load_source(image_path)

        result = {"source": source, "target": target}
        return result

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def load_item(self, path_item):
        return Image.open(path_item)

    def load_source(self, image_path):
        img = self.load_item(image_path)
        img = img.convert("RGB")
        source = self.transform(img)
        return source

    def load_source_cache(self, image_path):
        # cache = PyObjectCache()
        cache = MemStorageCache()
        image = cache.get(image_path)
        if image is None:
            image = self.load_item(image_path)
            cache.set(image_path, image)
            image = image.convert("RGB")
        source = self.transform(image)
        return source


if __name__ == "__main__":
    # ds = CheXpert_2_Dataset( #  256x256
    #     image_resize=(256, 256),
    #     augment_horizontal_flip=False,
    #     augment_vertical_flip=False,
    #     # path_root = '/home/Slp9280082/',
    #     path_root = '/mnt/d/chexpert',
    #     count=128,
    #     embedder_type=1 # RadBertEmbedder
    # )
    # for i in range(16):
    #     print(ds[i]['target'])
    ds = BreastCancerDataset(
        image_resize=(256, 256),
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root="/mnt/d/",
    )
    print(len(ds))
    for i in range(len(ds)):
        print(ds.labels[i][1])
