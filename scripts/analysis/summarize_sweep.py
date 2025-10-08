import csv,collections,statistics
path='outputs/sweep/sweep_results.csv'
rows=[]
with open(path, newline='', encoding='utf-8') as f:
    r=csv.DictReader(f)
    for rec in r:
        rows.append(rec)
configs=collections.defaultdict(list)
for rec in rows:
    configs[rec['config_name']].append(rec)
summary=[]
for cfg, recs in configs.items():
    def toint(x):
        if x in (None,'','NA'):
            return 0
        try:
            return int(x)
        except:
            return 0
    total_dets=[toint(x['total_detections']) for x in recs]
    persons=[toint(x['person_detections']) for x in recs]
    dup03=[toint(x['duplicate_pairs_iou_0.3']) for x in recs]
    dup05=[toint(x['duplicate_pairs_iou_0.5']) for x in recs]
    gtm03=[toint(x['gt_multi_match_iou_0.3']) for x in recs]
    gtm05=[toint(x['gt_multi_match_iou_0.5']) for x in recs]
    summary.append((cfg,len(recs),statistics.mean(total_dets),statistics.mean(persons),statistics.mean(dup03),statistics.mean(dup05),statistics.mean(gtm03),statistics.mean(gtm05)))
print('config,images,avg_total,avg_persons,avg_dup03,avg_dup05,avg_gtm03,avg_gtm05')
for s in sorted(summary, key=lambda x: (x[4],x[3])):
    print(f"{s[0]},{s[1]},{s[2]:.1f},{s[3]:.1f},{s[4]:.2f},{s[5]:.2f},{s[6]:.2f},{s[7]:.2f}")
print('\nTop 3 (by lowest avg_dup03):')
for s in sorted(summary, key=lambda x: (x[4],x[3]))[:3]:
    print(f"{s[0]} - avg_dup03={s[4]:.2f}, avg_persons={s[3]:.1f}, images={s[1]}")
