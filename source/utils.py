import time
import torch

import numpy as np

from skimage.transform import resize
from typing import List, Tuple


class ToTensor(object):
    def __call__(self, sample):
        positive_pair = sample['positive_pair']
        negative_pair = sample['negative_pair']
        
        size = positive_pair[0].shape
        new_size = (size[2], size[0], size[1])
        
        positive_pair = (torch.tensor(positive_pair[0], dtype=torch.float32).view(new_size),
                         torch.tensor(positive_pair[1], dtype=torch.float32).view(new_size))
        
        negative_pair = (torch.tensor(negative_pair[0], dtype=torch.float32).view(new_size),
                         torch.tensor(negative_pair[1], dtype=torch.float32).view(new_size))
        
        sample = {'positive_pair': positive_pair, 'negative_pair': negative_pair}
        
        return sample
        

class L1Norm(object):
    def __call__(self, first_representation: torch.Tensor, second_representation: torch.Tensor):
        norm = first_representation - second_representation
        norm = torch.norm(norm, 1, dim=1)
        norm = norm.mean()
        
        return norm
    
class CosineLoss(object):
    def __call__(self, first_representation: torch.Tensor, second_representation: torch.Tensor):
        assert len(first_representation.size()) == 2, 'invalid size for the first representation; the size should be [batch, dimension]'
        assert len(second_representation.size()) == 2, 'invalid size for the second representation; the size should be [batch, dimension]'
        assert first_representation.size(0) == second_representation.size(0), 'the batch size for both representations should be the same'
        assert first_representation.size(1) == second_representation.size(1), 'the dimensions of both representations should be the same'
        
        first_norm = first_representation.norm(dim=1)
        second_norm = second_representation.norm(dim=1)
        
        batch_size = first_representation.size(0)
        dimension = first_representation.size(1)
        
        similarity = torch.bmm(first_representation.view(batch_size, 1, dimension), 
                               second_representation.view(batch_size, dimension, 1)).view(-1)
        
        similarity = similarity / (first_norm * second_norm)
        # similarity = torch.arccos(similarity)
        
        return similarity


def cosine_similarity(first_representation: torch.Tensor, second_representation: torch.Tensor) -> torch.Tensor:
    assert len(first_representation.size()) == 2, 'invalid size for the first representation; the size should be [batch, dimension]'
    assert len(second_representation.size()) == 2, 'invalid size for the second representation; the size should be [batch, dimension]'
    assert first_representation.size(0) == second_representation.size(0), 'the batch size for both representations should be the same'
    assert first_representation.size(1) == second_representation.size(1), 'the dimensions of both representations should be the same'
    
    first_norm = first_representation.norm(dim=1)
    second_norm = second_representation.norm(dim=1)
    
    batch_size = first_representation.size(0)
    dimension = first_representation.size(1)
    
    similarity = torch.bmm(first_representation.view(batch_size, 1, dimension), 
                           second_representation.view(batch_size, dimension, 1)).view(-1)
    similarity = similarity / (first_norm * second_norm)
    
    return torch.abs(similarity)


def contrastive_loss(positive_pair: Tuple[torch.Tensor, torch.Tensor], negative_pairs: List[torch.Tensor], tau: torch.Tensor) -> torch.Tensor:
    """
    params:
    positive_pair   (Tuple[torch.Tensor, torch.Tensor]): a tuple containing the representations of the positive
        pairs. Each element of the positive_pair is a tensor of size [batch, dimension]
    negative_pairs  (List[torch.Tensor]): a list containing the representations of negative examples. Each
        element of the list is a tensor of size [batch, dimension]
    tau (torch.Tensor): the temperature variable as introduced in the SimCLR paper (this parameter is a scalar tensor)
    """
    
    nominator = torch.exp(cosine_similarity(*positive_pair) / tau)
    
    denominator = 0
    
    for item in negative_pairs:
        denominator += torch.exp(cosine_similarity(positive_pair[0], item) / tau)
        
    loss = nominator / denominator
    loss = loss.sum() / loss.size(0)
    
    return loss
    

def resize_image(image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    resized_image = resize(image, output_shape)
    
    return resized_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    return normalized_image


def centered_image(image: np.ndarray) -> np.ndarray:
    centered = image - np.mean(image)
    
    return centered


def random_homography_generator():
    np.random.seed(np.int64(time.time()))
    scale = 1
    rotation = 0.5 * np.pi * np.random.rand()
    translation = 128 * np.random.random((2,)) - 64
    
    homography = np.array([[scale * np.cos(rotation), -scale * np.sin(rotation), translation[0]], 
                           [scale * np.sin(rotation), scale * np.cos(rotation), translation[1]], 
                           [0, 0, 1]])
    
    return homography


def gaussian_kernel(value: np.float32, center: np.float32, variance: np.float32 = 1.) -> np.float32:
    coefficient = 1 / (np.sqrt(2 * np.pi * variance))
    kernel = np.exp(-(value - center) ** 2 / (2 * variance))
    
    # print(coefficient * kernel)
    
    return coefficient * kernel


def bilinear_kernel(value, center):
    return np.max([0, 1 - np.abs(center - value)])


def homography_solver(points, correspondents):
    coeff_matrix = np.zeros((2 * len(points), 9), dtype=np.float64)
    
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        xp = correspondents[i][0]
        yp = correspondents[i][1]
        coeff_matrix[i] = np.array([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        coeff_matrix[i + 1] = np.array([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    
    _, _, v = np.linalg.svd(coeff_matrix)
    
    return v[-1].reshape(3, 3)

