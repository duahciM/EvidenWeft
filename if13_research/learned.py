"""Fixed sparse XFeat ONNX export as a CPU-only candidate generator."""
import sys

import cv2
import numpy as np

from identity_catalog_matching.common import read,sha
from if12_research.contracts import reference

URL="https://github.com/DavideCatto/XFeat-ONNX/releases/download/V1.0.0/xfeat.onnx"
VERSION="1.23.2"


def backend(out,spec):
    lock=read(out/"model_backend_locked.json")
    file=out/"xfeat_sparse.onnx"
    if lock["url"]!=URL or lock["sha256"]!=sha(file) or file.stat().st_size!=2681450:
        raise RuntimeError("missing or different fixed XFeat export")
    sys.path.insert(0,str(out/"runtime_onnx"))
    import onnxruntime as ort
    if ort.__version__!=VERSION:
        raise RuntimeError(f"unexpected ONNX Runtime {ort.__version__}")
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=2
    opt.inter_op_num_threads=1
    opt.log_severity_level=3
    session=ort.InferenceSession(str(file),sess_options=opt,providers=["CPUExecutionProvider"])
    inp=session.get_inputs()[0]
    requested=["keypoints","descriptors","scores"]
    if {a.name for a in session.get_outputs()}!=set(requested):
        raise RuntimeError("unexpected XFeat feature export signature")
    fixed=inp.shape[-2:]
    observed_failures=[]
    def extract(frame):
        gray=frame["gray"]
        h,w=gray.shape
        hh,ww=(fixed[0] if isinstance(fixed[0],int) else h,fixed[1] if isinstance(fixed[1],int) else w)
        if (hh,ww)!=(h,w):
            gray=cv2.resize(gray,(ww,hh),interpolation=cv2.INTER_LINEAR)
        # Match the original grayscale evidence; color copies rather than adding a channel advantage.
        RGB=np.repeat(gray[None,None,:,:],3,axis=1).astype(np.float32)/255.0
        try:
            xy,desc,score=session.run(requested,{inp.name:RGB})
        except ort.capi.onnxruntime_pybind11_state.RuntimeException as exc:
            if "Can broadcast 0 by 0 or 1" not in str(exc):
                raise
            observed_failures.append({"sample":frame.get("sample"),"state":"feature_export_zero_keypoint_failure",
                                      "exception":str(exc)})
            return np.empty((0,2),np.float32),np.empty((0,64),np.float32),np.empty(0,np.float32)
        xy=np.asarray(xy,dtype=np.float32).reshape(-1,2)
        desc=np.asarray(desc,dtype=np.float32).reshape(-1,64)
        score=np.asarray(score,dtype=np.float32).ravel()
        if not (len(xy)==len(desc)==len(score)) or not all(np.isfinite(x).all() for x in (xy,desc,score)):
            raise ValueError("non-finite or misaligned learned output")
        xy=xy*np.float32([w/ww,h/hh])
        ordered=np.argsort(-score,kind="stable")[:spec["top_k"]]
        return xy[ordered],desc[ordered],score[ordered]
    fixture=np.random.default_rng(13).integers(0,256,size=(180,320),dtype=np.uint8)
    if not len(extract({"gray":fixture})[0]):
        raise RuntimeError("XFeat textured preflight returned no features")
    extract.failures=observed_failures
    return extract,{"method":"XFeat sparse ONNX V1.0.0","export":reference(file),"export_url":URL,
           "device":"CPU","runtime":ort.__version__,"input_shape":inp.shape,"top_k":spec["top_k"],
           "official_pytorch_parity_unverified":True,"same_proposal_confidence_gate":"reciprocal L2 ratio 0.75"}
