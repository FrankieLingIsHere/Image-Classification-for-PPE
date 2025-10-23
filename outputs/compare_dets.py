import json, sys
A='outputs/eval_rcnn_baseline_with_rescorer/evaluation_results_20251011_205308.json'
B='outputs/eval_rcnn_baseline_no_rescorer/evaluation_results_20251011_205439.json'
ja=json.load(open(A))
jb=json.load(open(B))
# Build map image->detections
def mkmap(j):
    m={}
    for item in j.get('detection_results',[]):
        img=item.get('image')
        dets=item.get('detections',[])
        m[img]=dets
    return m
ma=mkmap(ja)
mb=mkmap(jb)
imgs=sorted(set(ma.keys())|set(mb.keys()))
print('image,total_with,total_without,added,removed,changed_conf')
for img in imgs:
    da=ma.get(img,[])
    db=mb.get(img,[])
    # match by (class,bbox rounded)
    def key(d):
        cls=d.get('class')
        bb=d.get('bbox')
        # round bbox to 3 decimals if floats
        if bb and all(isinstance(x,float) for x in bb):
            bbk=tuple(round(x,3) for x in bb)
        else:
            bbk=tuple(int(round(x)) for x in bb) if bb else ()
        return (cls,bbk)
    ma_map={key(d):d for d in da}
    mb_map={key(d):d for d in db}
    keys_a=set(ma_map.keys())
    keys_b=set(mb_map.keys())
    added=len(keys_a-keys_b)
    removed=len(keys_b-keys_a)
    changed=0
    for k in (keys_a&keys_b):
        sa=ma_map[k].get('confidence',0)
        sb=mb_map[k].get('confidence',0)
        if abs(sa-sb)>1e-3 and round(sa,3)!=round(sb,3):
            changed+=1
    print(f"{img},{len(da)},{len(db)},{added},{removed},{changed}")
