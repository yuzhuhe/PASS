import torch
import torch.nn as nn
import torch.optim as optim
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import logging
import torch.nn.functional as F
from loader import *
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import model
import misc as sf #sf stands for supporting functions
from CLIP_guided.CLIP.clip import create_model
from CLIP_guided.CLIP.adapter import CLIP_Inplanted
from CLIP_guided.utils import augment, cos_sim, encode_text_with_prompt_ensemble
from CLIP_guided.prompt import REAL_NAME
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def consFn(x):
    y=torch.clamp(x,0,1)
    return y

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

fn= lambda x: sf.normalize01(np.abs(x))
#%
epochs = 300
nSave = 100
batchSz = 1
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(14)
if device.type == 'cuda':
    torch.cuda.manual_seed(14)
cuda = device
dir_data=[]



dir1 = '/home/ssddata/jufangmao/T1/train/Legion'
dir2 = '/home/ssddata/jufangmao/T1/train/NonLegion'
dir3 = '/home/ssddata/jufangmao/T1/train/study_label'
dir4 = '/home/ssddata/jufangmao/T1/eval'

AD_pth = 'fastMRI_T1/few-shot-12/Brain_T1.pth'

dir_data1 = make_dataset(dir1)
dir_data2 = make_dataset(dir2)
dir_data3 = make_dataset(dir3)
dir_data4 = make_dataset(dir4)
number_data = len(dir_data1) + len(dir_data2)
dir_data_1 = [dir1 + '/' + dir_data1[k] for k in range(len(dir_data1))]
dir_data_2 = [dir2 + '/' + dir_data2[k] for k in range(len(dir_data2))]
dir_data_3 = [dir3 + '/' + dir_data3[k] for k in range(len(dir_data3))]
dir_data.extend(dir_data_1)
dir_data.extend(dir_data_2)
dir_data.extend(dir_data_3)
number_data2 = len(dir_data4)

print ('*************************************************')
start_time = time.time()
checkpoint = 'fastMRI_T1_train_4_line_iter3/'
prepoint = 'train_pre/'

print('Data loading completed')

# Load anomaly detection model
clip_model_pre = create_model(model_name='ViT-L-14-336', img_size=320, device=device, pretrained='openai', require_pretrained=True)
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

del clip_model_pre
torch.cuda.empty_cache()

# Load memory bank
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
fewshot_norm_img = get_few_normal(dir2, 12, 0)
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
    'K': 3,  # iteration
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

'''Load Parameters'''
state_dict = model.state_dict()
pre_dict = torch.load(prepoint + 'acc4_line_pre.pth', map_location=torch.device('cpu'))
state_dict.update({k: v for k, v in pre_dict.items() if k in state_dict})
model.load_state_dict(state_dict)

pre_mask_para = pre_dict['create_mask1.ProbMask.mult']


model.to(device)
print('Model Parameters: ', num_params)

writer = SummaryWriter(comment='PASS_4_line')
params_to_train = [param for name, param in model.named_parameters() if (('Net_Mask' in name))]
optimizer=optim.Adam(params_to_train, lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=0.01, threshold_mode='rel', cooldown=10, min_lr=1e-6, eps=1e-08, verbose=False)


global_step2=0
print('training started at', datetime.now().strftime("%d-%b-%Y %I:%M "))
print('parameters are: Epochs:',epochs,' BS:',batchSz,'nSamples:',number_data//batchSz)

for epoch in range(epochs):
    model.train()
    with tqdm(total=number_data, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        Data_dir = dir_data
        Data_eval = dir_data4
        i=0
        for num_itr in range( number_data//batchSz):
            model.train()
            Sample = random.sample(Data_dir,batchSz)
            loader = getloader_dir_kspace(Sample)
            Data_dir = [x for x in Data_dir if x not in Sample]
            for a in loader:
                org_old, csm, _ = a
            org_old = org_old.to(device=device)
            csm = csm.to(device=device)

            # Reconstruction
            _, x_out1, Rec, AD_map, mask_loss, _ = model(org_old, csm, pre_mask_para.to(device=device))

            x_out1 = sf.torch_r2c(x_out1)
            Rec = sf.torch_r2c(Rec)

            tmp0 = torch.abs(x_out1 - org_old)
            tmp1 = torch.abs(Rec - org_old)
            tmp2 = torch.abs(AD_map * (Rec - org_old))

            batch_loss = 0.0
            loss0 = torch.mean(torch.pow(tmp0, 2))
            loss1 = torch.mean(torch.pow(tmp1, 2))
            loss2 = torch.mean(torch.pow(tmp2, 2))
            loss = loss0 + 2 * loss1 + loss2 + torch.abs(mask_loss)

            batch_loss += loss
            writer.add_scalar('Loss/batch_Loss', batch_loss.item(), epoch * (number_data//batchSz) + num_itr)

            batch_loss.backward()
            if (i+1)%8==0: # Multiple stacking to solve the problem of insufficient video memory
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.empty_cache()

            model.lamT1.data = consFn(model.lamT1.data)
            model.lamT2.data = consFn(model.lamT2.data)

            pbar.set_postfix(**{'loss (batch)': batch_loss.item()})
            pbar.update(org_old.shape[0])
            global_step2 += 1
            i += 1
        if (epoch % 10==0) or epoch==epochs-1:
            try:
                os.mkdir(checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            filtered_state_dict = {k: v for k, v in model.state_dict().items() if 'clip_model' not in k}
            torch.save(filtered_state_dict,
                       checkpoint + 'epoch_' + str(epoch) + '.pth')

            logging.info('Checkpoint saved !')
    j = 0
    psnr1 = 0
    psnr2 = 0
    '''Eval Section Begin'''
    model.to(device)
    model.eval()
    for i in tqdm(range(number_data2)):
        Sample = random.sample(Data_eval, 1)
        loader = getloader_dir_kspace(dir4, Sample)
        Data_eval = [x for x in Data_eval if x not in Sample]
        for aa in loader:
            org_old1, csm1, _ = aa
        _, x_out11, Rec1, _, _, _ = model(org_old1.to(device=device), csm1.to(device=device), pre_mask_para.to(device=device))
        psnr1 += sf.myPSNR(fn(org_old1.cpu().detach().numpy())[0], fn(sf.torch_r2c(x_out11).cpu().detach().numpy()))
        psnr2 += sf.myPSNR(fn(org_old1.cpu().detach().numpy())[0], fn(sf.torch_r2c(Rec1).cpu().detach().numpy()))

    writer.add_scalar('out1 PSNR', psnr1 / number_data2, epoch)
    writer.add_scalar('Rec PSNR', psnr2 / number_data2, epoch)
    print('PSNR1：', '{0:.3f}'.format(float(psnr1) / number_data2))
    print('PSNR2：', '{0:.3f}'.format(float(psnr2) / number_data2))
    scheduler.step(psnr2)
    if (float(psnr2) / number_data2) > 35:
        filtered_state_dict = {k: v for k, v in model.state_dict().items() if 'clip_model' not in k}
        torch.save(filtered_state_dict,
                       checkpoint + 'max_epoch_'+ str(epoch) + '.pth')
        print('save max acc or auc model')

    '''Eval Section End'''
###
end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('*************************************************')
writer.close()
