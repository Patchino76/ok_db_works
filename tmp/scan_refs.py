import json
import re

nb = json.load(open(r'c:\Projects\ok_db_works\pulse_db_transform\dataframe_test.ipynb', encoding='utf-8'))
pat = re.compile(r'DROP_FEATURES|Power_mean|ZumpfLevel|PumpRPM|spec_energy|amp_per_rpm')
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    for j, l in enumerate(''.join(c['source']).splitlines(), 1):
        if pat.search(l):
            print(f'cell {i:2} (id {c.get("id")}) line {j:3}: {l.strip()}')
