import cv2
import numpy as np


def candidates(gray,bg,cfg):
    mask=np.uint8(np.abs(gray-bg)>=cfg['delta'])
    if mask.mean()>cfg['max_component_area_fraction']:return [],'too_much_foreground_for_static_reference_proposal'
    n,ids,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    out=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<cfg['min_component_pixels'] or min(w,h)<2:continue
        cut=(ids[y:y+h,x:x+w]==k).astype(np.uint8)
        norm=cv2.resize(cut,(24,24),interpolation=cv2.INTER_NEAREST)>0
        appearance=cv2.resize(gray[y:y+h,x:x+w],(24,24))
        out.append({'bbox':[x,y,w,h],'center':centers[k].tolist(),'area':area,'shape':norm,'appearance':appearance})
    return out,'candidate_components' if out else 'no_candidate_component'


def dist_box(a,b):
    x,y,w,h=a; X,Y,W,H=b
    inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    return inter/max(1,min(w*h,W*H))


def public_candidate(k):
    return {x:y for x,y in k.items() if x not in ('shape','appearance')}
