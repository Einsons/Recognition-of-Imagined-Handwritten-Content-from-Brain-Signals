import os,sys
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models import DeepConvNet,EEGNet82
from src.extract import EEGDataset
from src.train_oof_aligned_dcn import align

WEIGHTS=np.asarray([5,3,3,9,0],dtype=np.float32);SCALES=np.asarray([1.,1.007047,1.128674,1.063061,1.160222],dtype=np.float32)
def predict(m,x,y,d):
    o=[];m.eval()
    with torch.no_grad():
        for bx,_ in DataLoader(EEGDataset(x,y),128):o.append(m(bx.to(d)).cpu())
    return torch.cat(o)
def main():
    cache_path=os.path.join(ROOT,'outputs','oof_multiarch_logits.npz')
    if not os.path.exists(cache_path):raise FileNotFoundError('Run src/evaluate_oof_multiarch_ensemble.py first to create the OOF cache.')
    cache=np.load(cache_path);oof=torch.tensor(cache['test']);target=torch.tensor(cache['test_targets']);oof=(oof*torch.tensor(SCALES)[None,:,None]*torch.tensor(WEIGHTS)[None,:,None]).sum(1)
    a=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));x=a['data'].astype('float32');y=a['labels_0indexed'];test=np.concatenate([np.flatnonzero(y==c)[270:] for c in range(26)]);raw=x[test,:,50:551];d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');root=os.path.join(ROOT,'models','checkpoints','full_refit')
    def dcn(name,z):
        m=DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5).to(d);m.load_state_dict(torch.load(os.path.join(root,name),map_location=d));return predict(m,z,y[test],d)
    def eeg(k,name):
        m=EEGNet82(24,26,input_time_points=801,temporal_kernel_length=k,dropout_rate=.3).to(d);m.load_state_dict(torch.load(os.path.join(root,name),map_location=d));return predict(m,x[test],y[test],d)
    dp=torch.stack([dcn(f'dcn_seed{s}.pth',raw) for s in (42,542)]).mean(0);ap=dcn('aligned_dcn_seed442.pth',align(raw));k25=eeg(25,'k25_seed142.pth');k15=torch.stack([eeg(15,f'k15_swa_seed{s}.pth') for s in (342,642)]).mean(0)
    full=5*dp+3*SCALES[1]*ap+3*SCALES[2]*k25+9*SCALES[3]*k15;full=full*(oof.std()/full.std());hybrid=(oof+full)/2
    print(f'OOF fold ensemble: {100*(oof.argmax(1)==target).float().mean().item():.2f}%');print(f'Full-development refit: {100*(full.argmax(1)==target).float().mean().item():.2f}%');print(f'Fixed 50/50 hybrid: {100*(hybrid.argmax(1)==target).float().mean().item():.2f}%')
if __name__=='__main__':main()
