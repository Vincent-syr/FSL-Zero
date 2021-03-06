import numpy as np
import os
import glob
import argparse
import backbone
import matplotlib.pyplot as plt
import torch

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet12 = backbone.ResNet12_func,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='ResNet10',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='protonet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_query'      , default=8, type=int,  help='number of unlabeled  query data in each class, same as n_query') #baseline and baseline++ only use this parameter in finetuning

    parser.add_argument('--train_aug'   , default=True, help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--n_episode', default=100, type=int, help = 'num of episodes in each epoch')
    parser.add_argument('--n_task', default=1, type=int, help = 'num of tasks in each episode')

    parser.add_argument('--mlp_dropout' , default=0.7, help='dropout rate in word embedding transformer')
    parser.add_argument('--attr_loss', default='BCE', help='focal_loss/BCE')
    # parser.add_argument('--aux'   , default=False,  help='use attribute as auxiliary data, multimodal method')

    # learning rate, optim
    parser.add_argument('--lr_anneal', default='const', help='const/pwc/exp, schedule learning rate')
    parser.add_argument('--init_lr', default=0.001)
    parser.add_argument('--optim', default='Adam', help='Adam/SGD')
    # learning rate decay
    
    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=300, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        # parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')

    elif script == 'train_gen':
        parser.add_argument('--save_freq'   , default=30, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=50, type=int, help ='Stopping epoch')    #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--miss_rate', default=0.4, help='missing rate of  attributes in validation')
    
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--source', default='feature', help='feature|image')
        parser.add_argument('--miss_rate', default=0, type=float,help='missing rate of  attributes in validation')

    else:
       raise ValueError('Unknown script')
        

    return parser.parse_args()


def get_trlog(params):
    trlog = {}
    trlog['script'] = 'pre-train'
    trlog['args'] = vars(params)
    trlog['epoch'] = []
    trlog['train_loss'] = []
    trlog['lr'] = []
    trlog['train_acc_attr'] = []
    trlog['val_acc_attr'] = []
    trlog['train_acc_cls'] = []
    trlog['val_acc_cls'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    return trlog


def get_trlog_vae(params):
    trlog = {}
    trlog['script'] = 'train_vae'
    trlog['args'] = vars(params)
    trlog['epoch'] = []
    trlog['train_loss'] = []
    trlog['lr'] = []
    trlog['syn_acc'] = []
    trlog['raw_acc'] = []
    trlog['miss_acc'] = []
    trlog['none_acc'] = []

    trlog['lambda_syn'] = []
    trlog['lambda_raw'] = []
    trlog['lambda_miss'] = []
    trlog['lambda_none'] = []
    
    return trlog

def get_trlog_test(params):
    trlog = {}
    trlog['script'] = 'test'
    trlog['args'] = vars(params)
    trlog['epoch'] = []
    trlog['base_acc'] = []
    trlog['val_acc'] = []
    trlog['novel_acc'] = []
    return trlog



def save_fig(trlog_path):
    trlog = torch.load(trlog_path)
    if 'script' not in trlog.keys():
        if 'syn_acc' in trlog.keys():
            trlog['script'] = 'train_vae'
        else:
            trlog['script'] = 'pre-train'

    print(trlog['script'])
    print(trlog['args'])

    if trlog['script'] == 'pre-train':
        print('max_acc = %.2f, max_acc_epoch = %d' % (trlog['max_acc'], trlog['max_acc_epoch']))
        train_loss = trlog['train_loss']
        train_acc_cls = trlog['train_acc_cls']
        val_acc_cls = trlog['val_acc_cls']
        train_acc_attr = trlog['train_acc_attr']
        val_acc_attr = trlog['val_acc_attr']
        # x = list(range(len(val_acc_cls)))
        x = trlog['epoch']

    # trlog['train_acc_attr'] = []
    # trlog['val_acc_attr'] = []
    # trlog['train_acc_cls'] = []
    # trlog['val_acc_cls'] = []

        plt.figure()
        l1, = plt.plot(x, train_loss,linewidth = 1.0)
        plt.title('Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig('%s_loss.jpg' % trlog_path)

        plt.figure()
        l1, = plt.plot(x, train_acc_cls, linewidth = 1.0)
        l2, = plt.plot(x, val_acc_cls, linewidth = 1.0)
        plt.title('Accuracy Class')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles = [l1, l2], labels=['train_acc_cls', 'val_acc_cls'],loc = 'best')
        plt.grid()
        plt.savefig('%s_acc_cls.jpg' % trlog_path)


        plt.figure()
        l1, = plt.plot(x, train_acc_attr, linewidth = 1.0)
        l2, = plt.plot(x, val_acc_attr, linewidth = 1.0)
        plt.title('Accuracy Attributes')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles = [l1, l2], labels=['train_acc_attr', 'val_acc_attr'],loc = 'best')
        plt.grid()
        plt.savefig('%s_acc_attr.jpg' % trlog_path)


        # plt.figure()
        # img_ratio = trlog['img_ratio']
        # lambda_c = trlog['lambda_c']
        # x = range(len(img_ratio))
        # plt.plot(x, img_ratio, linewidth = 1.0, label='img_ratio')
        # plt.plot(x, lambda_c, linewidth = 1.0, label='lambda')
        # plt.title('Image Ratio')
        # plt.xlabel('epoch')
        # plt.ylabel('Percentage')
        # plt.legend()
        # plt.grid()
        # plt.savefig('%s_img_ratio.jpg' % trlog_path)



        plt.figure()
        lr = trlog['lr']
        x = list(range(len(lr)))
        l1, = plt.plot(x, lr, linewidth = 1.0)
        plt.title('Learning rate')
        plt.xlabel('epoch')
        plt.ylabel('lr')
        plt.grid()
        plt.savefig('%s_params.jpg' % trlog_path)        



    elif trlog['script'] == 'train_vae':
        train_loss = trlog['train_loss']

        plt.figure()
        x = list(range(len(train_loss)))
        l1, = plt.plot(x, train_loss,linewidth = 1.0)
        plt.title('Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig('%s_loss.jpg' % trlog_path)

        plt.figure()
        x = list(range(len(trlog['syn_acc'])))
        l1, = plt.plot(x, trlog['syn_acc'], linewidth = 1.0)
        l2, = plt.plot(x, trlog['raw_acc'], linewidth = 1.0)
        l3, = plt.plot(x, trlog['miss_acc'], linewidth = 1.0)
        l4, = plt.plot(x, trlog['none_acc'], linewidth = 1.0)
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles = [l1,l2,l3,l4], labels=['syn_acc','raw_acc','miss_acc','none_acc'],loc = 'best')
        plt.grid()
        plt.savefig('%s_acc.jpg' % trlog_path)

        plt.figure()
        x = list(range(len(trlog['lambda_syn'])))
        l1, = plt.plot(x, trlog['lambda_syn'], linewidth = 1.0)
        l2, = plt.plot(x, trlog['lambda_raw'], linewidth = 1.0)
        l3, = plt.plot(x, trlog['lambda_miss'], linewidth = 1.0)
        l4, = plt.plot(x, trlog['lambda_none'], linewidth = 1.0)
        plt.title(r'$\lambda$ value during training')        #  r'$\Delta$rv' #对应于Δrv
        plt.xlabel('epoch')
        plt.ylabel(r'$\lambda$')
        plt.legend(handles = [l1,l2,l3,l4], labels=['lambda_syn','lambda_raw','lambda_miss','lambda_none'],loc = 'best')
        plt.grid()
        plt.savefig('%s_lambda.jpg' % trlog_path)


    elif trlog['script'] == 'test':
        acc = trlog['base_acc']
        x = list(range(len(acc)))
        l1, = plt.plot(x, trlog['base_acc'], linewidth = 1.0)
        l2, = plt.plot(x, trlog['val_acc'], linewidth = 1.0)
        l3, = plt.plot(x, trlog['novel_acc'], linewidth = 1.0)
        plt.title('Test Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles = [l1,l2,l3], labels=['base_acc','val_acc','novel_acc'],loc = 'best')      
        plt.grid()
        plt.savefig('%s_acc.jpg' % trlog_path)

    else:
        raise ValueError('Unknown Script !!')


def combine_trlog(trlog_list):
    pass





def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
