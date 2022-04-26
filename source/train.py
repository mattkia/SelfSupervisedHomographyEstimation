import torch
import cv2
import warnings

import torch.optim as optim
import torch.nn.functional as functional
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchgeometry.core import HomographyWarper
from sklearn.decomposition import PCA
from networks import Model, ModelInv
from utils import CosineLoss, L1Norm
from utils import centered_image, normalize_image, resize_image


warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_path = 'datasets'
image_size = 128
affine = False
enhanced = True
image_first = True
epochs = 100 if affine else 1000
inv = False
loss_id = 2
repr_dimension = 32

image1 = cv2.imread(data_path + '/1a.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(data_path + '/1b.jpg', cv2.IMREAD_GRAYSCALE)

image1 = resize_image(image1, (image_size, image_size))
image2 = resize_image(image2, (image_size, image_size))

image1 = normalize_image(image1)
image2 = normalize_image(image2)

image1 = centered_image(image1)
image2 = centered_image(image2)


timage1 = torch.tensor(image1, dtype=torch.float32).view(1, 1, image1.shape[0], image1.shape[1]).to(device)
timage2 = torch.tensor(image2, dtype=torch.float32).view(1, 1, image2.shape[0], image2.shape[1]).to(device)

print('Enhanced Mode' if enhanced else 'Regular Mode', '-', 
      'Affine Mode' if affine else 'Homography Mode', '-',
      'Image First' if image_first else 'Representation First', '-', 
      'Inverse Net' if inv else 'Regular Net - Running On', device)

if inv:
    model = ModelInv(in_channels=1, image_size=image_size, enhanced=True).to(device)
else:
    model = Model(in_channels=1, out_channels=repr_dimension, image_size=image_size, 
                  enhanced=enhanced, affine=affine, image_first=image_first).to(device)
    
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

loss_fn = L1Norm() if loss_id == 1 else CosineLoss()
ce_loss = nn.CrossEntropyLoss()

loss_values = []

for epoch in range(epochs):
    optimizer.zero_grad()
    repr1, repr2, warped_repr1, warped_repr2, homo2, homo1 = model(timage1, timage2)
    
    term1 = loss_fn(warped_repr1, repr2)
    term2 = loss_fn(warped_repr2, repr1)
    term3 = -torch.abs(loss_fn(repr1, repr2) - 3)
    term4 = -torch.abs(loss_fn(warped_repr1, warped_repr2) - 3)
    if affine:
        term5 = 0
        term6 = 0
        term7 = 0
        term8 = 0
    else:
        term5 = torch.norm(torch.bmm(homo1, homo2) - torch.eye(3).unsqueeze(0).to(device)) ** 2
        term6 = torch.norm(torch.bmm(homo2, homo1) - torch.eye(3).unsqueeze(0).to(device)) ** 2
        
        warper = HomographyWarper(image_size, image_size)
        warped_image1 = warper(timage1, homo1)
        warped_image2 = warper(timage2, homo2)
        
        term7 = ce_loss(warped_image1, timage2)
        term8 = ce_loss(warped_image2, timage1)
    
    loss = term1 + term2 + 0.03 * (term3 + term4) + term5 + term6 + term7 + term8
    
    loss_values.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        pca = PCA(n_components=3)
        v1 = repr1.detach().cpu().numpy()
        v2 = repr2.detach().cpu().numpy()
        v3 = warped_repr1.detach().cpu().numpy()
        v4 = warped_repr2.detach().cpu().numpy()
        vectors = np.array([v1, v2, v3, v4]).reshape(4, -1)
        

        print(f'[*] Epoch : {epoch + 1} - Loss Value : {loss.item():.3f}')
        
        new_vecs = pca.fit_transform(vectors)
        ax = plt.axes(projection='3d')
        for i in range(len(new_vecs)):
            if i % 2 == 0:
                ax.quiver(0, 0, 0, new_vecs[i][0], new_vecs[i][1], new_vecs[i][2], color='red')
            else:
                ax.quiver(0, 0, 0, new_vecs[i][0], new_vecs[i][1], new_vecs[i][2], color='green')
                
        plt.savefig(f'representations/frame{epoch}.jpg')
        plt.close()
        plt.imsave(f'results/frame{epoch}.jpg', warped_image1.detach().cpu().view(image_size, image_size).numpy())
    

homography = homo1

if affine:
    grid = functional.affine_grid(homography.detach().cpu(), timage1.size())
    warped_image = functional.grid_sample(timage1.cpu(), grid)
    warped_image = warped_image.view(image_size, image_size).numpy()
    print(homography)
else:
    warper = HomographyWarper(image_size, image_size)
    warped_image = warper(timage1.cpu(), homography.detach().cpu()).view(image_size, image_size).numpy()

    print(homography, torch.det(homography.squeeze(0)).item())

plt.subplot(1, 3, 1)
plt.imshow(image1)
plt.subplot(1, 3, 2)
plt.imshow(image2)
plt.subplot(1, 3, 3)
plt.imshow(warped_image)
plt.show()

np.savetxt(f'repr{repr_dimension}.txt', np.array(loss_values))

