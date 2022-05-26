import torch
import torch.nn as nn
from models.clip.prompt import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner

class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()
        cfg = cfgc()
        self.cfg = cfg
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.prompt_pool = None
        self.instance_prompt = None
        self.all_keys = None
        self.class_num = None

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def interface(self, image, selection):
        instance_batch = torch.stack([i.weight for i in self.instance_prompt], 0)[selection, :, :]
        image_features = self.image_encoder(image.type(self.dtype), instance_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for prompt in self.prompt_pool:
            tokenized_prompts = prompt.tokenized_prompts
            text_features = self.text_encoder(prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit
