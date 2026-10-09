"""xctrace export XML reader: resolves id/ref and returns rows as dicts mnemonic -> (fmt, text-or-value)."""
import xml.etree.ElementTree as ET
def read(path):
    root = ET.parse(path).getroot()
    schema = root.find('.//schema')
    cols = [c.findtext('mnemonic') for c in schema.findall('col')] if schema is not None else []
    ids = {}
    def reg(e):
        for d in e.iter():
            i = d.attrib.get('id')
            if i is not None: ids[i] = d
    out = []
    for row in root.iter('row'):
        vals = []
        for c in row:
            if 'ref' in c.attrib: c = ids[c.attrib['ref']]
            else: reg(c)
            vals.append(c)
        d = {}
        for k, v in zip(cols, vals):
            if v.tag == 'sentinel': d[k] = (None, None); continue
            txt = v.text
            if txt is None:
                # composite: collect first numeric/leaf text
                leaf = [x for x in v.iter() if x is not v]
                # resolve refs inside composites
                parts = []
                for x in leaf:
                    if 'ref' in x.attrib: x = ids[x.attrib['ref']]
                    if x.text: parts.append(x.text)
                txt = parts
            d[k] = (v.attrib.get('fmt'), txt)
        out.append(d)
    return out
