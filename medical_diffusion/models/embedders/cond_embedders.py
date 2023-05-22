
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer
from transformers import AutoTokenizer, BertModel


class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c



class MultiLabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(2**num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c
    

class RadBertEmbedder(nn.Module):
    _tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT")
    _model = BertModel.from_pretrained("StanfordAIMI/RadBERT")

    def __init__(self, emb_dim=32,*args, **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(768, emb_dim), # 768 bert output的维度
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, condition):
        inputs_list = [self._tokenizer(condition_str, return_tensors="pt") for condition_str in condition]
        outputs_list = [self._model(**inputs)for inputs in inputs_list]
        c = torch.stack([outputs.pooler_output[0].to('cuda') for outputs in outputs_list])
        c = self.mlp(c)
        return c
