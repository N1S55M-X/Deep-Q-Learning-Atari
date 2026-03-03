import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

frame_stack = deque(maxlen=4)

def preprocess(frame):
    frame = Image.fromarray(frame)
    return transform(frame)

def stack_frames(frame, new_episode):
    frame = preprocess(frame)

    if new_episode:
        frame_stack.clear()
        for _ in range(4):
            frame_stack.append(frame)
    else:
        frame_stack.append(frame)

    return torch.cat(list(frame_stack), dim=0)
