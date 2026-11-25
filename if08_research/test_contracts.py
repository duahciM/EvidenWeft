"""Tests for independently verifiable bridges and source-patch chronology."""
import unittest

import numpy as np

from .bridges import model_bridges,reach
from .chain import trace
from .run import CFG,PRE_CFG


class IF08Contracts(unittest.TestCase):
    def test_cross_model_validation_disallows_shared_source_and_detects_bad_reprojection(self):
        xy=np.float64([[10,10],[40,10],[60,12],[20,25],[45,27],[10,56],[40,54],[60,54]])
        def root(i,j,offset):
            return {'id':f'pair:{i}:{j}:SIFT','key':(i,j),'verified':True,'xy':(xy,xy+offset)}
        a,b,c=root(0,4,[1,0]),root(0,5,[1,0]),root(1,5,[1,0])
        pairs={v['key']:v for v in (a,b,c)}
        g=[{'windows':['M1','M2'],'root_ids':[a['id'],b['id'],c['id']]}]
        fs=[{'gray':np.zeros((64,64)),'t':i} for i in range(6)]
        models={'M1':{0:(np.eye(3),None,0),1:(np.eye(3),'l01',0)},'M2':{4:(np.eye(3),None,0),5:(np.eye(3),'r45',0)}}
        run=lambda: model_bridges(g,pairs,models,fs,PRE_CFG['R'],CFG['bridge'])
        base,test=run()
        self.assertEqual(base[0]['status'],'direct_frame_bridge_only')
        self.assertEqual(test[0]['validation_root'],c['id'])
        self.assertEqual(len(test[0]['evaluations']),1)
        c['xy']=(xy,xy+[12,0])
        self.assertEqual(run()[1][0]['status'],'holdout_not_supported')
        g[0]['root_ids']=[a['id'],b['id']]
        self.assertEqual(run()[1][0]['status'],'no_independent_holdout_pair')

    def test_sparse_reach_is_not_temporal_scene_merge(self):
        b=[{'status':'direct_frame_bridge_only','samples':[1,19],'pair':'pair:1:19:SIFT'}]
        raw={(0,1):None,(19,20):None}
        self.assertEqual(reach(0,20,raw,[],20,8)['status'],'unreachable_in_sampled_sparse_graph')
        self.assertEqual(reach(0,20,raw,b,20,8)['path'],['pair:0:1:SIFT','pair:1:19:SIFT','pair:19:20:SIFT'])

    def test_state_chain_tracks_a_real_lead_instead_of_imposing_first_frame(self):
        frames=[]
        for i in range(18):
            g=np.full((64,64),128,np.uint8)
            if i<4:g[25:32,25:32]=33
            elif i<10:g[25:32,25:32]=77
            else:g[25:32,25:32]=105
            frames.append({'gray':g,'sharpness':100.,'t':i*.2,'pts':i})
        m={'pair_metrics':[{'sample':i} for i in range(len(frames))]}
        p={'samples':[6,12],'area':49,'bbox_xywh':[25,25,7,7]}
        a=trace(frames,m,p,CFG['state'],len(frames))
        self.assertEqual(a['status'],'extended_registered_state_chain')
        self.assertEqual(a['samples'],[1,12])
        self.assertEqual(a['intermediate'],[6])

if __name__=='__main__':unittest.main()
