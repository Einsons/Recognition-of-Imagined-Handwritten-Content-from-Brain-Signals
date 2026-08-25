import os,sys
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.append(ROOT) if ROOT not in sys.path else None
from models import DeepConvNet,EEGNet82,apply_max_norm_constraints
from src.extract import EEGDataset
from src.train import set_seed
from src.train_oof_aligned_dcn import align

def train_model(model,x,y,epochs,seed,eeg=False,swa=False):
    set_seed(seed);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=model.to(d);loader=DataLoader(EEGDataset(x,y),64,shuffle=True);opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.05);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5);avg=None;n=0
    for epoch in range(1,epochs+1):
        model.train()
        for bx,by in loader:
            bx,by=bx.to(d),by.to(d)
            if eeg:
                bx=bx+torch.randn_like(bx)*.07;lam=np.random.beta(.2,.2);idx=torch.randperm(len(by),device=d);out=model(lam*bx+(1-lam)*bx[idx]);loss=lam*F.cross_entropy(out,by,label_smoothing=.1)+(1-lam)*F.cross_entropy(out,by[idx],label_smoothing=.1)
            else:out=model(bx);loss=F.cross_entropy(out,by)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if eeg:apply_max_norm_constraints(model)
        sched.step()
        if swa and epoch>=25:
            state=model.state_dict()
            if avg is None:avg={k:v.detach().clone() for k,v in state.items()};n=1
            else:
                for k in avg:avg[k]=(avg[k]*n+state[k].detach())/(n+1)
                n+=1
        if epoch%10==0:print(f'seed={seed} epoch={epoch}/{epochs}',flush=True)
    if avg is not None:model.load_state_dict(avg)
    return model
def pred(model,x,y):
    d=next(model.parameters()).device;model.eval();o=[]
    with torch.no_grad():
        for bx,_ in DataLoader(EEGDataset(x,y),128):o.append(model(bx.to(d)).cpu())
    return torch.cat(o)
def main():
    a=np.load(os.path.join(ROOT,'data','processed','eeg_dataset.npz'));x=a['data'].astype('float32');y=a['labels_0indexed'];dev=[];test=[]
    for c in range(26):idx=np.flatnonzero(y==c);dev.extend(idx[:270]);test.extend(idx[270:])
    dev=np.asarray(dev);test=np.asarray(test);raw=x[dev,:,50:551];rawt=x[test,:,50:551];outdir=os.path.join(ROOT,'models','checkpoints','full_refit');os.makedirs(outdir,exist_ok=True)
    dcn=[]
    for seed in (42,542):
        m=train_model(DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5),raw,y[dev],40,seed);torch.save(m.state_dict(),os.path.join(outdir,f'dcn_seed{seed}.pth'));dcn.append(pred(m,rawt,y[test]))
    ad=train_model(DeepConvNet(24,26,501,temporal_kernel=15,dropout_rate=.5),align(raw),y[dev],40,442);torch.save(ad.state_dict(),os.path.join(outdir,'aligned_dcn_seed442.pth'));adp=pred(ad,align(rawt),y[test])
    k25=train_model(EEGNet82(24,26,input_time_points=801,temporal_kernel_length=25,dropout_rate=.3),x[dev],y[dev],40,142,eeg=True);torch.save(k25.state_dict(),os.path.join(outdir,'k25_seed142.pth'));k25p=pred(k25,x[test],y[test])
    k15=[]
    for seed in (342,642):
        m=train_model(EEGNet82(24,26,input_time_points=801,temporal_kernel_length=15,dropout_rate=.3),x[dev],y[dev],60,seed,eeg=True,swa=True);torch.save(m.state_dict(),os.path.join(outdir,f'k15_swa_seed{seed}.pth'));k15.append(pred(m,x[test],y[test]))
    logits=5*torch.stack(dcn).mean(0)+3*1.007047*adp+3*1.128674*k25p+9*1.063061*torch.stack(k15).mean(0);target=torch.tensor(y[test]);print(f'FULL_REFIT_TEST={100*(logits.argmax(1)==target).float().mean().item():.2f}%')
if __name__=='__main__':main()
