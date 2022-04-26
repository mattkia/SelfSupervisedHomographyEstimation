import torch

import torch.nn as nn
import torch.nn.functional as functional

from torchgeometry.core import HomographyWarper
from torchvision import models



class SiameseNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, image_size):
        super().__init__()
        
        reduced_size = int(image_size / 32)

        self.encoder1 = nn.Conv2d(in_channels, 128, (3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.encoder2 = nn.Conv2d(128, 64, (3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.encoder3 = nn.Conv2d(64, 32, (3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.encoder4 = nn.Conv2d(32, 16, (3, 3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.encoder5 = nn.Conv2d(16, 3, (3, 3), stride=2, padding=1)
        
        self.flatten = nn.Flatten(start_dim=1)

        self.dense1 = nn.Linear(3 * reduced_size ** 2, 96)
        self.dense2 = nn.Linear(96, out_channels)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))

        x = self.flatten(x)

        x = torch.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class SiameseNetworkV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder1 = nn.Conv2d(in_channels, 16, (3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.encoder2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.encoder3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.encoder4 = nn.Conv2d(64, 128, (3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.encoder5 = nn.Conv2d(128, 128, (3, 3), stride=1, padding='same')

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))

        return x


class HomographyEstimator(nn.Module):
    def __init__(self, in_channels, image_size, affine=False):
        super().__init__()
        
        reduced_size = int(image_size / 32)

        self.encoder1 = nn.Conv2d(in_channels, 128, (3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.encoder2 = nn.Conv2d(128, 64, (3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.encoder3 = nn.Conv2d(64, 32, (3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.encoder4 = nn.Conv2d(32, 16, (3, 3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.encoder5 = nn.Conv2d(16, 3, (3, 3), stride=2, padding=1)
        
        self.flatten = nn.Flatten(start_dim=1)

        self.dense1 = nn.Linear(3 * reduced_size ** 2, 96)
        if affine:
            self.dense2 = nn.Linear(96, 6)
        else:
            self.dense2 = nn.Linear(96, 8)
        

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))
        
        x = self.flatten(x)
        
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class HomographyEstimatorV2(nn.Module):
    def __init__(self, in_channels, affine=False):
        super().__init__()
        
        self.dense1 = nn.Linear(in_channels, 128)
        self.dense2 = nn.Linear(128, 64)

        if affine:
            self.dense3 = nn.Linear(64, 6)
        else:
            self.dense3 = nn.Linear(64, 8)
        

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)

        return x
    

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, image_size: int, 
                 enhanced: bool=False, affine: bool=False, image_first: bool=False):
        super().__init__()
        
        self.__affine = affine
        self.__image_first = image_first
        
        if not affine:
            self.__warper = HomographyWarper(image_size, image_size)
        
        if enhanced:
            if image_first:
                self.__homography_estimator1 = EnhancedHomographyEstimator(2 * in_channels, image_size, affine)
                self.__homography_estimator2 = EnhancedHomographyEstimator(2 * in_channels, image_size, affine)
                self.__encoder1 = EnhancedSiameseNetwork(in_channels, out_channels, image_size)
                self.__encoder2 = EnhancedSiameseNetwork(in_channels, out_channels, image_size)
            else:
                self.__homography_estimator1 = HomographyEstimatorV2(2 * out_channels, affine)
                self.__homography_estimator2 = HomographyEstimatorV2(2 * out_channels, affine)
                self.__encoder1 = EnhancedSiameseNetwork(in_channels, out_channels, image_size)
                self.__encoder2 = EnhancedSiameseNetwork(in_channels, out_channels, image_size)
                
        else:
            if image_first:
                self.__homography_estimator1 = HomographyEstimator(2 * in_channels, image_size, affine)
                self.__homography_estimator2 = HomographyEstimator(2 * in_channels, image_size, affine)
                self.__encoder1 = SiameseNetwork(in_channels, out_channels, image_size)
                self.__encoder2 = SiameseNetwork(in_channels, out_channels, image_size)
            else:
                self.__homography_estimator1 = HomographyEstimatorV2(2 * out_channels, image_size, affine)
                self.__homography_estimator2 = HomographyEstimatorV2(2 * out_channels, image_size, affine)
                self.__encoder1 = SiameseNetwork(in_channels, out_channels, image_size)
                self.__encoder2 = SiameseNetwork(in_channels, out_channels, image_size)
                
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        representation1 = self.__encoder1(img1)
        representation2 = self.__encoder2(img2)
        
        
        if self.__image_first:
            image1 = torch.cat((img1, img2), dim=1)
            image2 = torch.cat((img2, img1), dim=1)
            homography1 = self.__homography_estimator1(image1)
            homography2 = self.__homography_estimator2(image2)
        else:
            repr1 = torch.cat((representation1, representation2), dim=1)
            repr2 = torch.cat((representation2, representation1), dim=1)
            
            homography1 = self.__homography_estimator1(repr1)
            homography2 = self.__homography_estimator2(repr2)
            
        if self.__affine:
            homography1 = homography1.view(-1, 2, 3)
            homography2 = homography2.view(-1, 2, 3)
            
            homography1 = homography1 / homography1[0, 1, 2]
            homography2 = homography2 / homography2[0, 1, 2]
            
            grid1 = functional.affine_grid(homography1, img1.size())
            warped_img1 = functional.grid_sample(img1, grid1)
            
            grid2 = functional.affine_grid(homography2, img2.size())
            warped_img2 = functional.grid_sample(img2, grid2)
        else:
            # homography1 = homography1.view(-1, 3, 3)
            # homography2 = homography2.view(-1, 3, 3)
        
            # homography1 = homography1 / homography1[0, 2, 2]
            # homography2 = homography2 / homography2[0, 2, 2]

            homo1 = torch.ones((1, 9), dtype=torch.float32).cuda()
            homo1[:, :8] = homography1
            homo1 = homo1.view(-1, 3, 3)
            
            homo2 = torch.ones((1, 9), dtype=torch.float32).cuda()
            homo2[:, :8] = homography2
            homo2 = homo2.view(-1, 3, 3)
            
            warped_img1 = self.__warper(img1, homo1)
            warped_img2 = self.__warper(img2, homo2)
        
        
        warped_representation1 = self.__encoder1(warped_img1)
        warped_representation2 = self.__encoder2(warped_img2)
        
        
        return representation1, representation2, warped_representation1, warped_representation2, homo2, homo1


class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.downsample1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=2, padding=1)
        self.downsample2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.downsample3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.downsample4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        
        self.upsample1 = nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(16, in_channels, (3, 3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(16)
        
    def forward(self, image):
        encoded1 = torch.relu(self.downsample1(image))
        encoded1 = self.bn1(encoded1)
        encoded2 = torch.relu(self.downsample2(encoded1))
        encoded2 = self.bn2(encoded2)
        encoded3 = torch.relu(self.downsample3(encoded2))
        encoded3 = self.bn3(encoded3)
        encoded4 = torch.relu(self.downsample4(encoded3))
        
        decoded = torch.relu(self.upsample1(encoded4, output_size=encoded3.size()))
        decoded = torch.relu(self.upsample2(decoded, output_size=encoded2.size()))
        decoded = torch.relu(self.upsample3(decoded, output_size=encoded1.size()))
        decoded = self.upsample4(decoded, output_size=image.size())
        
        return decoded, encoded4


class EnhancedSiameseNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, image_size):
        super().__init__()
        
        reduced_size = int(image_size / 32)

        self.encoder1 = nn.Conv2d(in_channels, 128, (3, 3), stride=1, padding='same')
        self.encoder1_b = nn.Conv2d(128, 128, (3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.encoder2 = nn.Conv2d(128, 64, (3, 3), stride=1, padding='same')
        self.encoder2_b = nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.encoder3 = nn.Conv2d(64, 32, (3, 3), stride=1, padding='same')
        self.encoder3_b = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.encoder4 = nn.Conv2d(32, 16, (3, 3), stride=1, padding='same')
        self.encoder4_b = nn.Conv2d(16, 16, (3, 3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.encoder5 = nn.Conv2d(16, 3, (3, 3), stride=1, padding='same')
        self.encoder5_b = nn.Conv2d(3, 3, (3, 3), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(3)
        
        self.flatten = nn.Flatten(start_dim=1)

        self.dense1 = nn.Linear(3 * reduced_size ** 2, 96)
        self.dense2 = nn.Linear(96, out_channels)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder1_b(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.encoder2_b(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = torch.relu(self.encoder3_b(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = torch.relu(self.encoder4_b(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))
        x = self.bn5(x)
        x = torch.relu(self.encoder5_b(x))

        x = self.flatten(x)

        x = torch.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class EnhancedSiameseNetworkV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder1 = nn.Conv2d(in_channels, 16, (3, 3), stride=1, padding='same')
        self.encoder1_b = nn.Conv2d(16, 16, (3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.encoder2 = nn.Conv2d(16, 32, (3, 3), stride=1, padding='same')
        self.encoder2_b = nn.Conv2d(32, 32, (3, 3), stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.encoder3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding='same')
        self.encoder3_b = nn.Conv2d(64, 64, (3, 3), stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.encoder4 = nn.Conv2d(64, 128, (3, 3), stride=1, padding='same')
        self.encoder4_b = nn.Conv2d(128, 128, (3, 3), stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.encoder5 = nn.Conv2d(128, 128, (3, 3), stride=1, padding='same')
        self.encoder5_b = nn.Conv2d(128, 128, (3, 3), stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder1_b(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.encoder2_b(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = torch.relu(self.encoder3_b(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = torch.relu(self.encoder4_b(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))
        x = self.bn5(x)
        x = torch.relu(self.encoder5_b(x))

        return x


class EnhancedHomographyEstimator(nn.Module):
    def __init__(self, in_channels, image_size, affine=False):
        super().__init__()
        
        reduced_size = int(image_size / 32)

        self.encoder1 = nn.Conv2d(in_channels, 128, (3, 3), stride=1, padding='same')
        self.encoder1_b = nn.Conv2d(128, 128, (3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.encoder2 = nn.Conv2d(128, 64, (3, 3), stride=1, padding='same')
        self.encoder2_b = nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.encoder3 = nn.Conv2d(64, 32, (3, 3), stride=1, padding='same')
        self.encoder3_b = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.encoder4 = nn.Conv2d(32, 16, (3, 3), stride=1, padding='same')
        self.encoder4_b = nn.Conv2d(16, 16, (3, 3), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.encoder5 = nn.Conv2d(16, 3, (3, 3), stride=1, padding='same')
        self.encoder5_b = nn.Conv2d(3, 3, (3, 3), stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(3)
        
        self.flatten = nn.Flatten(start_dim=1)

        self.dense1 = nn.Linear(3 * reduced_size ** 2, 96)
        if affine:
            self.dense2 = nn.Linear(96, 6)
        else:
            self.dense2 = nn.Linear(96, 8)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder1_b(x))
        x = self.bn1(x)
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.encoder2_b(x))
        x = self.bn2(x)
        x = torch.relu(self.encoder3(x))
        x = torch.relu(self.encoder3_b(x))
        x = self.bn3(x)
        x = torch.relu(self.encoder4(x))
        x = torch.relu(self.encoder4_b(x))
        x = self.bn4(x)
        x = torch.relu(self.encoder5(x))
        x = self.bn5(x)
        x = torch.relu(self.encoder5_b(x))

        x = self.flatten(x)

        x = torch.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class ModelInv(nn.Module):
    def __init__(self, in_channels: int, image_size: int, enhanced: bool=False):
        super().__init__()

        self.__warper = HomographyWarper(image_size, image_size)
        
        if enhanced:
            self.__encoder1 = EnhancedSiameseNetworkV2(in_channels)
            self.__encoder2 = EnhancedSiameseNetworkV2(in_channels)
            self.__homography_estimator = EnhancedHomographyEstimator(256, image_size)
        else:
            self.__encoder1 = SiameseNetworkV2(in_channels)
            self.__encoder2 = SiameseNetworkV2(in_channels)
            self.__homography_estimator = HomographyEstimator(2 * in_channels, image_size)
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        representation1 = self.__encoder1(img1)
        representation2 = self.__encoder2(img2)
        
        repr1 = torch.cat([representation1, representation2], dim=1)
        repr2 = torch.cat([representation2, representation1], dim=1)
        
        homography1 = self.__homography_estimator(repr1)
        homography2 = self.__homography_estimator(repr2)
        
        homography1 = homography1.view(-1, 3, 3)
        homography2 = homography2.view(-1, 3, 3)
        
        homography1 = homography1 / homography1[0, 2, 2]
        homography2 = homography2 / homography2[0, 2, 2]
        
        warped_img1 = self.__warper(img1, homography1)
        warped_img2 = self.__warper(img2, homography2)
        
        warped_representation1 = self.__encoder1(warped_img1)
        warped_representation2 = self.__encoder2(warped_img2)
        
        
        return representation1, representation2, warped_representation1, warped_representation2, homography2, homography1
