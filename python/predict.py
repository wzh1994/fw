import torch

from PIL import Image
from sklearn.externals import joblib
import glob
import os
import torch
import torchvision.transforms as transforms
from interface import interface, FireworkType
import multiprocessing
from conv3d import conv3d


def pil_loader(img_str):
    with Image.open(img_str) as img:
        img = img.convert('RGB')
        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img)
    return img


def _get_item(pic_type):
    file_path = os.path.join('extract', pic_type)
    images = sorted(glob.glob(os.path.join(file_path, '*jpg')))
    img = torch.stack([pil_loader(image) for image in images], 1)
    return img


def get_eval_item():
    img1 = _get_item('original')
    img2 = _get_item('difference')
    img = torch.cat([img1[:, :, 33 : 366, 33 : 366], img2[:, :, 33 : 366, 33 : 366]])
    return torch.stack([img]).cuda()


def get_model(model_name, num_args):
    print("=> start predicting")
    model = conv3d(num_args)
    model.cuda()
    load_path = os.path.join('models', '{}.pth.tar'.format(model_name))
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('missing keys from checkpoint {}: {}'.format(model_name, k))
        print("=> loaded model from checkpoint '{}'".format(load_path))
        model.eval()
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_name))
    load_path = os.path.join('models', '{}.scalar.pkl'.format(model_name))
    if os.path.isfile(load_path):
        scalar = joblib.load(load_path)
        print("=> loaded scalar from checkpoint '{}'".format(load_path))
    else :
        raise RuntimeError("=> no scalar file found at '{}'".format(model_name))
    return model, scalar


def _predict(fname, model_name, num_args, q):
    interface.extract(fname)
    model, scalar = get_model(model_name, num_args)
    y = model(get_eval_item()).detach().cpu().numpy()
    print("=> predict done")
    r = scalar.inverse_transform(y)
    q.put(r[0])


def type_analysis(type):
    if type == FireworkType.Normal:
        return 'normal', 449
    elif type == FireworkType.MultiExplosion:
        return 'multi', 452
    elif type == FireworkType.Circle:
        return 'circle', 453
    elif type == FireworkType.Twinkle:
        return 'twinkle', 498
    elif type == FireworkType.DualMixture:
        return 'dualmix', 898
    else:
        raise RuntimeError('Not supported type')


def predict(fname, type):
    model_name, num_args = type_analysis(type)
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_predict, args=(fname, model_name, num_args, q))
    p.start()
    p.join()
    return list(q.get())


if __name__ == '__main__':
    print(predict('11.avi', FireworkType.MultiExplosion))
