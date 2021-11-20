import torch
import cv2 as cv
import numpy as np

from train import Generator, Discriminator

model_name = 'epochs10'

seed = 100
torch.manual_seed(seed)

count = 10
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f'Device => {device}')

netG = Generator(ngpu=ngpu).to(device)
netG.load_state_dict(torch.load(f'models/{model_name}_g.pth'))
netD = Discriminator(ngpu=ngpu).to(device)
netD.load_state_dict(torch.load(f'models/{model_name}_d.pth'))

gridsize, image_size = 6, 64
fixed_noise = torch.randn(gridsize*gridsize, nz, 1, 1, device=device)
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu().numpy()

# for i in range(count):
#     img = np.transpose(fake[i], (1, 2, 0)).copy()
#     cv.imshow('img', cv.resize(img, (500, 500)))
#     cv.waitKey()
# cv.destroyAllWindows()

blank = np.zeros((image_size*gridsize, image_size*gridsize, 1))
for i in range(6*6):
    blank[i%gridsize*image_size:(i%gridsize+1)*image_size, int(i/gridsize)*image_size:int(i/gridsize+1)*image_size] = \
        np.transpose(fake[i], (1, 2, 0))
cv.imwrite(f'generated{gridsize}x{gridsize}.jpg', blank*255)
cv.imshow('img', blank)
cv.waitKey()
cv.destroyAllWindows()