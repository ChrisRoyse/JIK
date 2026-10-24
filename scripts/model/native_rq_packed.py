"""Packed equal-atom loss for the qualified joint no-commitment experiment.

Every instrument and window remains typed. Packing is across independent atoms
with the same window layout; it changes reduction arithmetic, not the objective.
"""
import torch
from torch.nn import functional as F

from model.native_rq_batch import BatchedInterventionNativeRQ
from text.constellation_corpus import need


class PackedNativeRQ(BatchedInterventionNativeRQ):
    def __init__(self,dimensions,config,sizes,arm,statistics):
        need(arm=='no_encoder_quantizer','RQ_PACKED_ARM')
        super().__init__(dimensions,config,sizes,arm,statistics)

    def set_step(self,number):
        need(type(number) is int and number>=0,'RQ_PACKED_CURSOR')
        self.step_number=number

    def pack(self,frames):
        need(isinstance(frames,(list,tuple)) and frames,'RQ_PACKED_EMPTY')
        device=self.encoder.query.device;counts=[]
        for frame in frames:
            need(set(frame)==set(self.roster),'RQ_PACKED_ROSTER');row={}
            for name,dims in self.dimensions.items():
                x=frame[name]
                need(isinstance(x,torch.Tensor) and x.ndim==2 and x.shape[1]==dims and len(x)>0
                     and x.dtype==torch.float32 and x.device==device and not x.requires_grad,'RQ_PACKED_NATIVE')
                row[name]=len(x)
            counts.append(row)
        groups=self.groups(counts);packed=[]
        for sizes,indices in groups:
            native={n:torch.stack([frames[i][n] for i in indices]) for n in self.roster}
            need(all(bool(torch.isfinite(v).all()) for v in native.values()),'RQ_PACKED_NONFINITE')
            packed.append((sizes,indices,native))
        return packed

    def encode_packed(self,packed,count):
        encoder=self.encoder;device=encoder.query.device
        atom_position=encoder.position(torch.zeros(1,dtype=torch.int64,device=device))[0]
        result=[None]*count
        for sizes,indices,native in packed:
            tokens=[];queries=[];cursor=0
            for lane,(name,windows) in enumerate(zip(self.roster,sizes)):
                normalized=(native[name]-getattr(self,'center_'+str(lane)))/getattr(self,'variance_'+str(lane)).sqrt()
                local=encoder.position(torch.arange(windows,device=device))
                tokens.append(self.projectors[name](normalized)+encoder.kind[lane]+atom_position+local)
                cursor+=windows;queries.append(cursor)
                query=encoder.query[lane]+encoder.kind[lane]+atom_position
                tokens.append(query[None,None].expand(len(indices),1,-1));cursor+=1
            x=torch.cat(tokens,1);mask=torch.zeros((cursor,cursor),dtype=torch.bool,device=device)
            for block in encoder.blocks:x=block(x,mask)
            chosen=torch.tensor(queries,dtype=torch.int64,device=device)
            hidden=encoder.norm(x)[:,chosen];values=self.to_latent(hidden.reshape(len(indices),-1))
            for i,value in zip(indices,values):result[i]=value
        return torch.stack(result)

    def latent(self,frames):
        return self.encode_packed(self.pack(frames),len(frames))

    def decode_group(self,values,lane,name,windows):
        positions=self.encoder.position(torch.arange(windows,device=values.device))
        joined=torch.cat((values[:,None].expand(-1,windows,-1),positions[None].expand(len(values),-1,-1)),-1)
        return self.baseline(lane,windows)+getattr(self,'variance_'+str(lane)).sqrt()*self.decoders[name](joined)

    def training_loss(self,frames,donors,eligible):
        packed=self.pack(frames);latent=self.encode_packed(packed,len(frames));codes,values,penalty=self.quantize(latent)
        need(bool(torch.isfinite(values).all()),'RQ_PACKED_LATENT')
        need(donors.shape==eligible.shape==(len(frames),) and donors.dtype==torch.int64
             and eligible.dtype==torch.bool and donors.device==eligible.device==latent.device
             and bool(((donors>=0)&(donors<len(frames))).all()),'RQ_PACKED_DONORS')
        any_eligible=bool(eligible.any());errors={n:[] for n in self.roster};contrasts={n:[] for n in self.roster}
        decoded=[{} for _ in frames]
        for sizes,indices,native in packed:
            active=eligible[indices];use_contrast=any_eligible and bool(active.any())
            z=values[indices];wrong=values[donors[indices]] if use_contrast else None
            for lane,(name,windows) in enumerate(zip(self.roster,sizes)):
                right=self.decode_group(z,lane,name,windows)
                mse=(right-native[name]).square().mean(dim=(1,2));errors[name].append(mse)
                if use_contrast:
                    bad=self.decode_group(wrong,lane,name,windows)
                    delta=(mse-(bad-native[name]).square().mean(dim=(1,2)))/getattr(self,'variance_'+str(lane))
                    contrasts[name].append(F.softplus(delta[active]))
                for index,value in zip(indices,right):decoded[index][name]=value
        lanes={n:torch.cat(errors[n]).mean() for n in self.roster}
        reconstruction=torch.stack([lanes[n]/getattr(self,'variance_'+str(i)) for i,n in enumerate(self.roster)]).mean()
        contrast=torch.stack([torch.cat(contrasts[n]).mean() for n in self.roster]).mean() if any_eligible else reconstruction*0
        return dict(loss=reconstruction+contrast+penalty,reconstruction=reconstruction,contrast=contrast,
                    quantizer_loss=penalty,regularizer=reconstruction*0,stage='quantized_joint',latent=latent,
                    values=values,codes=codes,decoded=tuple(decoded),per_lane=lanes)
