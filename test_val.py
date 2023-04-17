from cmath import e
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms as tfs
import cv2
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from model import DB_Net
from sklearn import metrics

colormap =[[0,0,0],[128,0,0]]
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i

def image2label(label_im):
    data = cv2.cvtColor(label_im, cv2.COLOR_BGR2RGB)
    data = np.array(data,dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx],dtype='int64')

def cal_cm(y_true,y_pred):
    y_true=y_true.reshape(1,-1).squeeze()
    y_pred=y_pred.reshape(1,-1).squeeze()
    #print(y_true,y_pred)
    cm=metrics.confusion_matrix(y_true,y_pred)
    return cm

def Intersection_over_Union(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    IoU = intersection / union
    return IoU

def colormap_smoke(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([128, 0, 0]) 
    return cmap

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg

def Colorize(gray_image):
    cmap=colormap_smoke(2)
    size = gray_image.shape  
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    for label in range(0, len(cmap)):  
        mask = gray_image[0] == label
        color_image[0][mask] = cmap[label][0]  
        color_image[1][mask] = cmap[label][1]  
        color_image[2][mask] = cmap[label][2] 
    return color_image


if __name__ == '__main__':
    # **************************__model__***********************
    num_classes = 2
    model = DB_Net(num_classes)
    pthfile = 'pthfile_path/Coarse2fine.pth' #coarse、fine、fine2coarse、coarse2fine
    model.load_state_dict(torch.load(pthfile, map_location='cpu'))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Trainable Params: ', trainable_params)

    image_path="/PoVSSeg/image/"
    label_path="/PoVSSeg/label/"
    with open("/PoVSSeg/Smoke_splits/val.txt", "r") as f:  
        data = f.readlines()  
    background=[]
    smoke=[]
    for i in data:
        i=i.strip('\n')
        frame=image_path+"img"+i.split("el")[1]+'.png'
        label=label_path+i+'.png'

        frame=cv2.imread(frame)
        frame = cv2.resize(frame, (960, 540))  # 960,540
        im_tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        frame = im_tfs(frame)
        frame = Variable(torch.unsqueeze(frame, dim=0).float(), requires_grad=False)
        frame = frame.to(device)
        outputs = model(frame)
        outputs = F.softmax(outputs, dim=1)
        smoke_output = outputs.max(dim=1)[1].data.cpu().numpy()
        smoke_output = np.squeeze(smoke_output)

        lable = cv2.imread(label)
        lable = cv2.resize(lable, (960, 540),interpolation=cv2.INTER_NEAREST)
        lable = image2label(lable)
        lable = lable[::16, ::16]

        eval_cm = cal_cm(smoke_output, lable)
        eval_iou = Intersection_over_Union(eval_cm)
        eval_background_iu = eval_iou[0]
        eval_smoke_iu = eval_iou[1]
        print(i,eval_background_iu,eval_smoke_iu)
        background.append(eval_background_iu)
        smoke.append(eval_smoke_iu)

    print('mean IOU of background: ',sum(background)/len(background))
    print('mean IOU of smoke: ',sum(smoke)/len(smoke))
    


