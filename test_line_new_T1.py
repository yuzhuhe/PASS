import torch
import torch.nn.functional as F
import os,time
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import model
import matplotlib.pyplot as plt
from tqdm import tqdm
import misc as sf #sf stands for supporting functions
import random
from loader import *
import pytorch_ssim
from CLIP_guided.CLIP.clip import create_model
from CLIP_guided.CLIP.adapter import CLIP_Inplanted
from CLIP_guided.utils import augment, cos_sim, encode_text_with_prompt_ensemble
from CLIP_guided.prompt import REAL_NAME

import warnings
warnings.filterwarnings('ignore')

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(fname)
            images.append(path)
    return images
def get_few_normal(dataset_path, shot, iterate):
    x = []
    img_dir = os.path.join(dataset_path)
    normal_names = os.listdir(img_dir)

    # select images
    if iterate < 0:
        random_choice = random.sample(normal_names, shot)
    else:
        random_choice = []
        with open(f'T1-12.txt', 'r', encoding='utf-8') as infile:
            for line in infile:
                data_line = line.strip("\n").split()
                if data_line[0] == f'n-{iterate}:':
                    random_choice = data_line[1:]
                    break

    for f in random_choice:
        if f.endswith('.npz'):
            x.append(os.path.join(img_dir, f))

    fewshot_img = []
    for idx in range(shot):
        image = x[idx]

        image = np.abs(np.load(image, 'r')['org'])
        image = torch.from_numpy(image)
        image = image.repeat(3, 1, 1)

        # image = Image.open(image).convert('RGB')
        # image = self.transform_x(image)
        fewshot_img.append(image.unsqueeze(0))

    fewshot_img = torch.cat(fewshot_img)
    return fewshot_img
def init_model(params):
    '''
    M, N, acc_factor, scale, lam1, lam2, sigma, cgIter, cgTol, ADMM_Iter, ADMM_tol, ADMM_rho, K,
    mask_loss1, mask_loss2, lr, AD_pth, clip_model, text_features, seg_mem_features
    '''
    return model.model_line(params['M'], params['N'], params['acc_factor'], params['scale'],
                                 params['lam1'], params['lam2'], params['sigma'], params['cgIter'], params['cgTol'],
                                 params['ADMM_Iter'], params['ADMM_tol'], params['ADMM_rho'], params['K'],
                                 params['mask_loss1'], params['mask_loss2'], params['lr'], params['clip_model'],
                                 params['text_features'], params['seg_mem_features'])
def consFn(x):
    y = torch.clamp(x, 0, 1)
    return y

fn= lambda x: sf.normalize01(np.abs(x))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(14)
if device.type == 'cuda':
    torch.cuda.manual_seed(14)

epochs=100
nSave=100
batchSz=1
lr=1e-3
dir_checkpoint='checkpoint/'
prepoint = 'checkpoint/'


dir1 = 'test_ex'
dir_data = make_dataset(dir1)
dir_data = [dir1 + '/' + dir_data[k] for k in range(len(dir_data))]

nImg = len(dir_data)
AD_pth = 'fastMRI_T1/few-shot-12/Brain_T1.pth'
clip_model_pre = create_model(model_name='ViT-L-14-336', img_size=320, device=device,
                          pretrained='openai', require_pretrained=True)
clip_model_pre.eval()
AD_model = CLIP_Inplanted(clip_model=clip_model_pre, features=[6, 12, 18, 24]).to(device)
AD_model.eval()

# Load pre trained model
AD_checkpoint = torch.load(AD_pth)
AD_model.seg_adapters.load_state_dict(AD_checkpoint["seg_adapters"])
AD_model.det_adapters.load_state_dict(AD_checkpoint["det_adapters"])

# Load text_feature
with torch.cuda.amp.autocast(), torch.no_grad():
    text_features = encode_text_with_prompt_ensemble(clip_model_pre, REAL_NAME['Brain'], device)

# Load memory bank
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
fewshot_norm_img = get_few_normal('memory_bank', 12, 0)
augment_normal_img, augment_normal_mask = augment(fewshot_norm_img)
support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

# Load seg_mem_features
seg_features = []
for image in support_loader:
    image = image[0].to(device)
    with torch.no_grad():
        _, seg_patch_tokens, _ = AD_model(image)
        seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
        seg_features.append(seg_patch_tokens)
seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in
                    range(len(seg_features[0]))]

# Model parameter
model_params = {
    'M': 320,
    'N': 320,
    'acc_factor': 0.17,
    'scale': 0.08,
    'sigma': 0.0,
    'lam1': 0.015,
    'lam2': 0.015,
    'K': 3,
    'cgIter': torch.tensor(5),
    'cgTol': torch.tensor(1e-12, dtype=torch.float32),
    'ADMM_Iter': torch.tensor(5),
    'ADMM_tol': torch.tensor(1e-6, dtype=torch.float32),
    'ADMM_rho': torch.tensor(1),
    'mask_loss1': 1.0,
    'mask_loss2': 0.5,
    'lr': lr,
    'clip_model': AD_model,
    'text_features': text_features,
    'seg_mem_features': seg_mem_features
}

model = init_model(model_params)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model Parameters: ', num_params)

mask_shape = torch.zeros(1, 320, 320, 1)
mask1, mask2, _ = model.create_mask1(mask_shape)
init_mask = consFn(mask1 + mask2)

state_dict = model.state_dict()
pre_dict = torch.load(dir_checkpoint + 'acc_line_4.pth', map_location=torch.device('cpu'))
state_dict.update({k: v for k, v in pre_dict.items() if k in state_dict})
model.load_state_dict(state_dict)

# pre_mask_para = pre_dict['create_mask1.ProbMask.mult']
pre_mask_para = torch.load(prepoint + 'acc_line_4_pre.pth',
                        map_location=torch.device('cpu'))['create_mask1.ProbMask.mult'].to(device)

psnr1 = []
psnr2 = []
psnr3 = []
num_bounding = 0
SSIM1 = []
SSIM2 = []
SSIM3 = []
model.to(device)
model.eval()
for i in tqdm(range(nImg)):
    Sample = [dir_data[0]]
    loader,Legion = getloader_dir1_classify(Sample)
    dir_data = [x for x in dir_data if x not in Sample]
    for a in loader:
        org_old,csm, label,n,nx,ny,nx_length,ny_length = a

    csm = csm.to(device=device)
    org_old = org_old.to(device=device)
    with torch.no_grad():
        atb, x_out1, Rec, AD_map, _, mask = model(org_old, csm, pre_mask_para)

    # torch.cuda.empty_cache()

    CAM = AD_map.cpu().detach().numpy()
    CAM = sf.normalize01(CAM)
    normOrg = fn(org_old.cpu().detach().numpy())
    normatb = fn(atb.cpu().detach().numpy())
    normRec1 = fn(sf.torch_r2c(x_out1).cpu().detach().numpy()) #未加强
    normRec2 = fn(sf.torch_r2c(Rec).cpu().detach().numpy())  # 加强


    psnr1.append(sf.myPSNR(normOrg[0], normRec1))
    psnr2.append(sf.myPSNR(normOrg[0], normRec2))

    SSIM1.append(pytorch_ssim.ssim(torch.from_numpy(normOrg).to(torch.float32), torch.from_numpy(normRec1[np.newaxis,:]).to(torch.float32)))
    SSIM2.append(pytorch_ssim.ssim(torch.from_numpy(normOrg).to(torch.float32), torch.from_numpy(normRec2[np.newaxis,:]).to(torch.float32)))

    #bounding box
    for kkk in range(int(n)):
        psnr3.append(sf.myPSNR(normOrg[0,0,int(nx[0, kkk]):int(nx[0, kkk])+int(nx_length[0, kkk]),int(ny[0, kkk]):int(ny[0, kkk])+int(ny_length[0, kkk])],
                           normRec2[0,int(nx[0, kkk]):int(nx[0, kkk])+int(nx_length[0, kkk]),int(ny[0, kkk]):int(ny[0, kkk])+int(ny_length[0, kkk])]))
        SSIM3.append(pytorch_ssim.ssim(torch.from_numpy(
            normOrg[0:1, 0:1, int(nx[0, kkk]):int(nx[0, kkk]) + int(nx_length[0, kkk]),
            int(ny[0, kkk]):int(ny[0, kkk]) + int(ny_length[0, kkk])]).to(torch.float32),
                                   torch.from_numpy(normRec2[np.newaxis, 0:1,
                                                    int(nx[0, kkk]):int(nx[0, kkk]) + int(nx_length[0, kkk]),
                                                    int(ny[0, kkk]):int(ny[0, kkk]) + int(ny_length[0, kkk])]).to(
                                       torch.float32)))

        num_bounding += 1


    # visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(init_mask[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Init_Mask")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(mask[0, :, :, 0].detach().cpu().numpy(), cmap='gray')
    axes[1, 0].set_title("Learned_Mask")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(np.abs(normOrg[0, 0, :]), cmap='gray')
    axes[0, 1].set_title("Original")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(np.abs(normRec2[0, :]), cmap='gray')
    axes[1, 1].set_title(
        "Reconstruction\nPSNR: " + str(format(float(sf.myPSNR(normOrg, normRec2)), ".3f"))
    )
    axes[1, 1].axis("off")


    axes[0, 2].imshow(np.abs(normOrg[0, 0, :]), cmap='gray', alpha=1)
    ax_cam = axes[0, 2]

    # Add bounding boxes
    for ii in range(int(n)):
        ax_cam.add_patch(
            plt.Rectangle(
                (int(nx[0, ii]), int(ny[0, ii])),
                int(nx_length[0, ii]),
                int(ny_length[0, ii]),
                color="red",
                fill=False,
                linewidth=1
            )
        )

    # Overlay CAM
    axes[0, 2].imshow(CAM[0, 0, :, :], alpha=0.3, interpolation='nearest', cmap='jet')
    axes[0, 2].set_title("Original + CAM")
    axes[0, 2].axis("off")


    axes[1, 2].imshow(np.abs(normRec2[0, :]), cmap='gray', alpha=1)
    ax_cam = axes[1, 2]

    # Add bounding boxes
    for ii in range(int(n)):
        ax_cam.add_patch(
            plt.Rectangle(
                (int(nx[0, ii]), int(ny[0, ii])),
                int(nx_length[0, ii]),
                int(ny_length[0, ii]),
                color="red",
                fill=False,
                linewidth=1
            )
        )

    # Overlay CAM
    axes[1, 2].imshow(CAM[0, 0, :, :], alpha=0.3, interpolation='nearest', cmap='jet')
    axes[1, 2].set_title("Reconstruction + CAM")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

psnr3 = [value for value in psnr3 if value > 0]

print('Average PSNR (without enhancement):', np.mean(psnr1))
print('Average SSIM (without enhancement):', np.mean(SSIM1))
print('PSNR standard deviation (without enhancement):', np.std(psnr1))
print('SSIM standard deviation (without enhancement):', np.std(SSIM1))

print('Average PSNR:', np.mean(psnr2))
print('Average SSIM:', np.mean(SSIM2))
print('PSNR standard deviation:', np.std(psnr2))
print('SSIM standard deviation:', np.std(SSIM2))

print('PSNR inside bounding box:', '{0:.3f}'.format(np.mean(psnr3)))
print('SSIM inside bounding box:', '{0:.3f}'.format(np.mean(SSIM3)))
print('PSNR standard deviation inside bounding box:', '{0:.3f}'.format(np.std(psnr3)))
print('SSIM standard deviation inside bounding box:', '{0:.3f}'.format(np.std(SSIM3)))

