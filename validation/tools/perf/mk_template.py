"""mk_template.py <src form.template> <out.tracetemplate> key=value ...  (values: int, true/false)
Re-points entries of the recording-control-state dict (counterscounterprofile, countersshaderprofiler, ...) to new objects."""
import plistlib, sys
src, dst, *kv = sys.argv[1:]
d = plistlib.load(open(src, 'rb'))
o = d['$objects']
r = lambda x: o[x.data] if isinstance(x, plistlib.UID) else x
target = None
for i, x in enumerate(o):
    if isinstance(x, dict) and 'NS.keys' in x and 'counterscounterprofile' in [r(k) for k in x['NS.keys']]:
        target = x
assert target is not None
keys = [r(k) for k in target['NS.keys']]
for item in kv:
    k, v = item.split('=')
    val = {'true': True, 'false': False}.get(v.lower())
    if val is None: val = int(v)
    o.append(val)
    uid = plistlib.UID(len(o) - 1)
    if k in keys: target['NS.objects'][keys.index(k)] = uid
    else:
        o.append(k); target['NS.keys'].append(plistlib.UID(len(o) - 1)); target['NS.objects'].append(uid)
    print(f'{k} = {val!r}')
plistlib.dump(d, open(dst, 'wb'), fmt=plistlib.FMT_BINARY, sort_keys=False)
