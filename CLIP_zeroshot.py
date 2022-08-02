import json
import os
import argparse
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.core50_data_loader import CORE50

deepfake = ['real', 'fake']

corenames = [
    "plug adapter 1",
    "plug adapter 2",
    "plug adapter 3",
    "plug adapter 4",
    "plug adapter 5",
    "mobile phone 1",
    "mobile phone 2",
    "mobile phone 3",
    "mobile phone 4",
    "mobile phone 5",
     "scissor 1",
     "scissor 2",
     "scissor 3",
     "scissor 4",
     "scissor 5",
     "light bulb 1",
     "light bulb 2",
     "light bulb 3",
     "light bulb 4",
     "light bulb 5",
     "can 1",
     "can 2",
     "can 3",
     "can 4",
     "can 5",
     "glass 1",
     "glass 2",
     "glass 3",
     "glass 4",
     "glass 5",
     "ball 1",
     "ball 2",
     "ball 3",
     "ball 4",
     "ball 5",
     "marker 1",
     "marker 2",
     "marker 3",
     "marker 4",
     "marker 5",
     "cup 1",
     "cup 2",
     "cup 3",
     "cup 4",
     "cup 5",
     "remote control 1",
     "remote control 2",
     "remote control 3",
     "remote control 4",
     "remote control 5",
]

domainnet_classnames = [
    "aircraft carrier",
    "airplane",
    "alarm_clock",
    "ambulance",
    "angel",
    "animal_migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball_bat",
    "basket",
    "basketball",
    "bat",
    "bathtub",
    "beach",
    "bear",
    "beard",
    "bed",
    "bee",
    "belt",
    "bench",
    "bicycle",
    "binoculars",
    "bird",
    "birthday_cake",
    "blackberry",
    "blueberry",
    "book",
    "boomerang",
    "bottlecap",
    "bowtie",
    "bracelet",
    "brain",
    "bread",
    "bridge",
    "broccoli",
    "broom",
    "bucket",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "calendar",
    "camel",
    "camera",
    "camouflage",
    "campfire",
    "candle",
    "cannon",
    "canoe",
    "car",
    "carrot",
    "castle",
    "cat",
    "ceiling fan",
    "cello",
    "cell_phone",
    "chair",
    "chandelier",
    "church",
    "circle",
    "clarinet",
    "clock",
    "cloud",
    "coffee_cup",
    "compass",
    "computer",
    "cookie",
    "cooler",
    "couch",
    "cow",
    "crab",
    "crayon",
    "crocodile",
    "crown",
    "cruise_ship",
    "cup",
    "diamond",
    "dishwasher",
    "diving_board",
    "dog",
    "dolphin",
    "donut",
    "door",
    "dragon",
    "dresser",
    "drill",
    "drums",
    "duck",
    "dumbbell",
    "ear",
    "elbow",
    "elephant",
    "envelope",
    "eraser",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fence",
    "finger",
    "fire_hydrant",
    "fireplace",
    "firetruck",
    "fish",
    "flamingo",
    "flashlight",
    "flip_flops",
    "floor_lamp",
    "flower",
    "flying_saucer",
    "foot",
    "fork",
    "frog",
    "frying_pan",
    "garden",
    "garden_hose",
    "giraffe",
    "goatee",
    "golf_club",
    "grapes",
    "grass",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "headphones",
    "hedgehog",
    "helicopter",
    "helmet",
    "hexagon",
    "hockey_puck",
    "hockey_stick",
    "horse",
    "hospital",
    "hot_air_balloon",
    "hot_dog",
    "hot_tub",
    "hourglass",
    "house",
    "house_plant",
    "hurricane",
    "ice_cream",
    "jacket",
    "jail",
    "kangaroo",
    "key",
    "keyboard",
    "knee",
    "knife",
    "ladder",
    "lantern",
    "laptop",
    "leaf",
    "leg",
    "light_bulb",
    "lighter",
    "lighthouse",
    "lightning",
    "line",
    "lion",
    "lipstick",
    "lobster",
    "lollipop",
    "mailbox",
    "map",
    "marker",
    "matches",
    "megaphone",
    "mermaid",
    "microphone",
    "microwave",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mountain",
    "mouse",
    "moustache",
    "mouth",
    "mug",
    "mushroom",
    "nail",
    "necklace",
    "nose",
    "ocean",
    "octagon",
    "octopus",
    "onion",
    "oven",
    "owl",
    "paintbrush",
    "paint can",
    "palm tree",
    "panda",
    "pants",
    "paper_clip",
    "parachute",
    "parrot",
    "passport",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "piano",
    "pickup_truck",
    "picture_frame",
    "pig",
    "pillow",
    "pineapple",
    "pizza",
    "pliers",
    "police car",
    "pond",
    "pool",
    "popsicle",
    "postcard",
    "potato",
    "power outlet",
    "purse",
    "rabbit",
    "raccoon",
    "radio",
    "rain",
    "rainbow",
    "rake",
    "remote control",
    "rhinoceros",
    "rifle",
    "river",
    "roller coaster",
    "rollerskates",
    "sailboat",
    "sandwich",
    "saw",
    "saxophone",
    "school bus",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea turtle",
    "see saw",
    "shark",
    "sheep",
    "shoe",
    "shorts",
    "shovel",
    "sink",
    "skateboard",
    "skull",
    "skyscraper",
    "sleeping bag",
    "smiley_face",
    "snail",
    "snake",
    "snorkel",
    "snowflake",
    "snowman",
    "soccer_ball",
    "sock",
    "speedboat",
    "spider",
    "spoon",
    "spreadsheet",
    "square",
    "squiggle",
    "squirrel",
    "stairs",
    "star",
    "steak",
    "stereo",
    "stethoscope",
    "stitches",
    "stop sign",
    "stove",
    "strawberry",
    "streetlight",
    "string bean",
    "submarine",
    "suitcase",
    "sun",
    "swan",
    "sweater",
    "swing set",
    "sword",
    "syringe",
    "table",
    "teapot",
    "teddy-bear",
    "telephone",
    "television",
    "tennis racquet",
    "tent",
    "The_Eiffel_Tower",
    "The Great Wall of China",
    "The_Mona_Lisa",
    "tiger",
    "toaster",
    "toe",
    "toilet",
    "tooth",
    "toothbrush",
    "toothpaste",
    "tornado",
    "tractor",
    "traffic light",
    "train",
    "tree",
    "triangle",
    "trombone",
    "truck",
    "trumpet",
    "t-shirt",
    "umbrella",
    "underwear",
    "van",
    "vase",
    "violin",
    "washing_machine",
    "watermelon",
    "waterslide",
    "whale",
    "wheel",
    "windmill",
    "wine_bottle",
    "wine_glass",
    "wristwatch",
    "yoga",
    "zebra",
    "zigzag",
]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

"""
python eval.py --dataroot ~/workspace/datasets/DeepFake_Data/CL_data/ --datatype deepfake 
python eval.py --dataroot ~/workspace/datasets/domainnet --datatype domainnet 
python eval.py --dataroot ~/workspace/datasets/core50/data/core50_128x128/ --datatype core50      

"""

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--dataroot', type=str, default='/home/wangyabin/workspace/DeepFake_Data/CL_data/',
                        help='data path')
    parser.add_argument('--datatype', type=str, default='core50', help='data type')
    return parser


class DummyDataset(Dataset):
    def __init__(self, data_path, data_type):

        self.trsf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        labels = []
        if data_type == "deepfake":
            subsets = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
            multiclass = [0, 0, 0, 0, 0]
            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, 'val')
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else ['']
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        images.append(os.path.join(root_, cls, '0_real', imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        images.append(os.path.join(root_, cls, '1_fake', imgname))
                        labels.append(1 + 2 * id)
        elif data_type == "domainnet":
            self.data_root = data_path
            self.image_list_root = self.data_root
            self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in
                                self.domain_names]
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                image_list = open(image_list_path).readlines()
                imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]

            for item in imgs:
                images.append(os.path.join(self.data_root, item[0]))
                labels.append(item[1])
        elif data_type == "core50":
            self.dataset_generator = CORE50(root=data_path, scenario="ni")
            images, labels = self.dataset_generator.get_test_set()
            labels = labels.tolist()
        else:
            pass

        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(self.pil_loader(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

import clip

model, preprocess = clip.load("ViT-B/16")

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights



zeroshot_weights = zeroshot_classifier(domainnet_classnames, imagenet_templates)
# zeroshot_weights = zeroshot_classifier(corenames, imagenet_templates)
# zeroshot_weights = zeroshot_classifier(deepfake, imagenet_templates)

args = setup_parser().parse_args()
device = "cuda:0"
model = model.to(device)
test_dataset = DummyDataset(args.dataroot, args.datatype)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
from tqdm import tqdm

top1, n = 0., 0.
for path, inputs, targets in tqdm(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        image_features = model.encode_image(inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = accuracy(logits, targets%345, topk=(1, 5))
        # acc1, acc5 = accuracy(logits, targets%50, topk=(1, 5))
        # acc1, acc5 = accuracy(logits, targets%2, topk=(1, 2))
        top1 += acc1
        n += inputs.size(0)



top1 = (top1 / n) * 100


print(top1)