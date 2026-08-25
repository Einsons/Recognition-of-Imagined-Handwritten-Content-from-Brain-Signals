import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:sys.path.append(ROOT)

from models import DeepConvNet,EEGNet82,GraphEEGNet
from src.extract import EEGDataset
from src.train_oof_aligned_dcn import align
from src.train_oof_dcn import make_oof_splits

NAMES=('dcn','aligned_dcn','eegnet_k25','eegnet_k15_swa','graph')
DCN_SEED_WEIGHTS=(.50,.50)
K15_SEED_WEIGHTS=(.50,.50)


def predict(model,data,labels,device):
    out=[];model.eval()
    with torch.no_grad():
        for x,_ in DataLoader(EEGDataset(data,labels),batch_size=128):out.append(model(x.to(device)).cpu())
    return torch.cat(out)


def seed_average(models,data,labels,device,weights):
    return sum(weight*predict(model,data,labels,device)
               for model,weight in zip(models,weights))


def compositions(total,count,prefix=()):
    if count==1:
        yield prefix+(total,);return
    for value in range(total+1):yield from compositions(total-value,count-1,prefix+(value,))


def main():
    arc=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));data=arc['data'].astype(np.float32);labels=arc['labels_0indexed']
    splits,test_idx=make_oof_splits(labels,5);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oof=[[] for _ in NAMES];tests=[[] for _ in NAMES];targets=[]
    for fold,(_,val_idx) in enumerate(splits):
        dcns=[]
        for seed in (42,542):
            dcn=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(device);dcn.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_dcn_0_2000',f'fold_{fold}_seed_{seed+fold}.pth'),map_location=device));dcns.append(dcn)
        aligned=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(device);aligned.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_aligned_dcn',f'fold_{fold}_seed_{442+fold}.pth'),map_location=device))
        k25=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=25,dropout_rate=.3).to(device);k25.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_eegnet_k25',f'fold_{fold}_seed_{142+fold}.pth'),map_location=device))
        k15s=[]
        for seed in (342,642):
            k15=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=15,dropout_rate=.3).to(device);k15.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_eegnet_k15_swa',f'fold_{fold}_seed_{seed+fold}.pth'),map_location=device));k15s.append(k15)
        graph=GraphEEGNet().to(device);graph.load_state_dict(torch.load(os.path.join(ROOT,'models','checkpoints','oof_graph',f'fold_{fold}_seed_{242+fold}.pth'),map_location=device))
        raw_val=data[val_idx,:,50:551];raw_test=data[test_idx,:,50:551]
        targets.append(torch.from_numpy(labels[val_idx]).long())
        oof[0].append(seed_average(dcns,raw_val,labels[val_idx],device,DCN_SEED_WEIGHTS));tests[0].append(seed_average(dcns,raw_test,labels[test_idx],device,DCN_SEED_WEIGHTS))
        inputs=((aligned,align(raw_val),align(raw_test)),(k25,data[val_idx],data[test_idx]))
        for index,(model,val_x,test_x) in enumerate(inputs,start=1):
            oof[index].append(predict(model,val_x,labels[val_idx],device));tests[index].append(predict(model,test_x,labels[test_idx],device))
        oof[3].append(seed_average(k15s,data[val_idx],labels[val_idx],device,K15_SEED_WEIGHTS));tests[3].append(seed_average(k15s,data[test_idx],labels[test_idx],device,K15_SEED_WEIGHTS))
        oof[4].append(predict(graph,raw_val,labels[val_idx],device));tests[4].append(predict(graph,raw_test,labels[test_idx],device))
        print(f'Loaded fold {fold+1}/5',flush=True)
    oof=[torch.cat(x) for x in oof];target=torch.cat(targets);raw_test=[torch.stack(x).mean(0) for x in tests]
    cache_dir=os.path.join(ROOT,'outputs');os.makedirs(cache_dir,exist_ok=True)
    np.savez(os.path.join(cache_dir,'oof_multiarch_logits.npz'),
             oof=np.stack([x.numpy() for x in oof],axis=1),
             test=np.stack([x.numpy() for x in raw_test],axis=1),
             oof_targets=target.numpy(),test_targets=labels[test_idx],
             fold_ids=np.repeat(np.arange(5),len(splits[0][1])))
    scales=[oof[0].std()/x.std() for x in oof];oof=[x*s for x,s in zip(oof,scales)]
    best_accuracy,best_weights=0,None
    for weights in compositions(20,len(NAMES)):
        logits=sum(w*x for w,x in zip(weights,oof));acc=100*(logits.argmax(1)==target).float().mean().item()
        if acc>best_accuracy:best_accuracy,best_weights=acc,weights
    test_logits=[x*s for x,s in zip(raw_test,scales)];test_target=torch.from_numpy(labels[test_idx]).long()
    test_accuracy=100*(sum(w*x for w,x in zip(best_weights,test_logits)).argmax(1)==test_target).float().mean().item()
    print('Architectures:',', '.join(NAMES));print('OOF scales:',', '.join(f'{float(x):.6f}' for x in scales));print('OOF-selected integer weights:',best_weights)
    print(f'OOF accuracy: {best_accuracy:.2f}%');print(f'Held-out test accuracy: {test_accuracy:.2f}%')


if __name__=='__main__':main()
