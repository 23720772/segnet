import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from metrics import iou_score, dice_coef, recall_s, precision_s, accuracy_s, jaccard_coef, f1_score_s,specificity_s,sensitivity_s,auc_score,mcc_score
import archs
from dataset import Dataset
from metrics import iou_score, dice_coef, recall_s, precision_s, accuracy_s, jaccard_coef
from utils import AverageMeter
"""                 
需要指定参数：--name dsb2018_96_NestedUNet_woDS dsb2018_96_UNet_woDS dsb2018_96_SegNet_woDS
dsb2018_96_DFANet_woDS   dsb2018_96_GRUUNet_woDS "dsb2018_96_D_UNet_woDS"
"""
# DFANet  to C:\Users\孙浩然/.cache\torch\hub\checkpoints\resnet101-5d3b4d8f.pth

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="dsb2018_96_SegNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'FPN':
        blocks = [2, 4, 23, 3]
        model = archs.__dict__[config['arch']](blocks, config['num_classes'], back_bone='resnet101')

    elif config['arch'] == 'SegNet':
        model = archs.__dict__[config['arch']](config['num_classes'])
    elif config['arch'] == 'AttU_Net':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'TransUnet':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'TransUnet2':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'GRUUNet':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'RUNet':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'RUNet2':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'D_UNet':
        model = archs.__dict__[config['arch']](in_channels=3, num_classes=config['num_classes'])
    elif config['arch'] == 'CUNet':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == 'DFANet':
        ch_cfg = [[8, 48, 96],
                  [240, 144, 288],
                  [240, 144, 288]]
        model = archs.__dict__[config['arch']](ch_cfg,64,1)
    elif config['arch'] == 'SETR':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'], image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048)
    elif config['arch'] == 'SETR2':
        model = archs.__dict__[config['arch']](num_classes=config['num_classes'], image_size=512, patch_size=512//16, dim=1024, depth = 24, heads = 16, mlp_dim = 2048)
    elif config['arch'] == 'Double_UNet':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'build_doubleunet':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'C2FNet':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'MENet':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'TwinLiteNet':
        model = archs.__dict__[config['arch']]()
    elif config['arch'] == 'DCSAU_Net':
        model = archs.__dict__[config['arch']]()
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision']
                                               )

    #
    # if config['arch']=='FPN':
    #     blocks = [2, 4, 23, 3]
    #     model = archs.__dict__[config['arch']](blocks, config['num_classes'], back_bone='resnet101')
    #
    # elif config['arch']=='SegNet':
    #     model = archs.__dict__[config['arch']](config['num_classes'])
    #
    # elif config['arch']=='DeepLabV3Plus':
    #     model = archs.__dict__[config['arch']](config['num_classes'])
    # elif config['arch'] == 'VisionTransformer':
    #     model = archs.__dict__[config['arch']](img_size=512, num_classes=1, zero_head=False, vis=False)
    # elif config['arch'] == 'AttU_Net':
    #     model = archs.__dict__[config['arch']]()
    # else:
    #     model = archs.__dict__[config['arch']](config['num_classes'],
    #                                            config['input_channels'],
    #                                            config['deep_supervision']
    #                                            )



    model = model.cuda()

    # Data loading code

    # img_ids = glob(os.path.join('inputs',config['dataset'], 'images', '*' + config['img_ext']))

    img_ids = glob(os.path.join('inputs', "test", 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids = img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], )

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', "test", 'images'),
        mask_dir=os.path.join('inputs', "test", 'masks'),

        # img_dir=os.path.join('inputs', config['dataset'], 'images'),
        # mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        # img_dir="inputs\\test\\images",
        #
        # mask_dir="inputs\\test\\masks",
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],

        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    avg_dice_meter = AverageMeter()

    avg_recall_meter = AverageMeter()
    avg_precision_meter = AverageMeter()
    avg_accuracy_meter = AverageMeter()
    avg_f1_score_meter = AverageMeter()
    avg_specificity_meter = AverageMeter()
    avg_sensitivity_meter = AverageMeter()
    avg_auc_meter = AverageMeter()
    avg_mcc_meter = AverageMeter()

    avg_jaccard_meter = AverageMeter()


    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()



            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            # output = output[-1]

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            # compute dice score
            dice = dice_coef(output, target)
            avg_dice_meter.update(dice, input.size(0))

            recall = recall_s(output, target)
            avg_recall_meter.update(recall, input.size(0))

            precision = precision_s(output, target)
            avg_precision_meter.update(precision, input.size(0))

            accuracy = accuracy_s(output, target)
            avg_accuracy_meter.update(accuracy, input.size(0))

            f1_score = f1_score_s(output, target)
            avg_f1_score_meter.update(f1_score, input.size(0))
            specificity = specificity_s(output, target)
            avg_specificity_meter.update(specificity, input.size(0))
            sensitivity = sensitivity_s(output, target)
            avg_sensitivity_meter.update(sensitivity, input.size(0))

            auc = auc_score(output, target)
            avg_auc_meter.update(auc, input.size(0))
            mcc = mcc_score(output, target)
            avg_mcc_meter.update(mcc, input.size(0))

            jaccard = jaccard_coef(output, target)
            avg_jaccard_meter.update(jaccard, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    print('Dice: %.4f' % avg_dice_meter.avg)
    print('jaccard: %.4f' % avg_jaccard_meter.avg)
    print('recall: %.4f' % avg_recall_meter.avg)
    print('precision: %.4f' % avg_precision_meter.avg)
    print('accuracy: %.4f' % avg_accuracy_meter.avg)

    print('sensitivity: %.4f' % avg_sensitivity_meter.avg)
    print('specificity: %.4f' % avg_specificity_meter.avg)
    print('auc: %.4f' % avg_auc_meter.avg)
    print('mcc: %.4f' % avg_mcc_meter.avg)

    print('f1_score: %.4f' % avg_f1_score_meter.avg)

    
    plot_examples(input, target, model,num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        # image_arr = model(datax[image_indx:image_indx+1])[-1].squeeze(0).detach().cpu().numpy()
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()

        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
