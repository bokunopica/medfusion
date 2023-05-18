import copy
import torch 
import torch.nn as nn 
import os
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

class EMAModel(nn.Module):
    # See: https://github.com/huggingface/diffusers/blob/3100bc967084964480628ae61210b7eaa7436f1d/src/diffusers/training_utils.py#L42  
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        super().__init__()
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.averaged_model = self.averaged_model #.to(device=model.device)

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param

            if not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.averaged_model.load_state_dict(ema_state_dict, strict=False)
        self.optimization_step += 1



class PyObjectCache(object):
    """
    cache for training data img
    """
    _instance = None
    __cache_space = None

    def __new__(cls):
        if PyObjectCache._instance is None:
            cls._instance = super().__new__(cls)
            cls.__cache_space = dict()
        return cls._instance
    
    def set(self, key, value):
        self.__cache_space[key] = copy.deepcopy(value)

    def get(self, key):
        return self.__cache_space.get(key)
    
    

class WebCache(object):
    """
    cache for training data img in a web app
    """
    def set(self, key, value):
        
        pass

    def get(self, key):
        pass


class MemStorageCache(object):
    """
    cache for training data img in a web app
    """
    _path = "/run/user/10009"
    def set(self, img_path: str, image: Image):
        key_path = f"{self._path}/{img_path}"
        dir_path = '/'.join(key_path.split('/')[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image.save(key_path)

    def get(self, img_path:str):
        key_path = f"{self._path}/{img_path}"
        if os.path.exists(key_path):
            image = Image.open(key_path).convert('RGB')
        else:
            image = None
        return image
    


if __name__ == "__main__":
    a, b = PyObjectCache(), PyObjectCache()
    a.set('test', b)
    print(a)
    print(b)
    print(b.get('test'))