import xml.etree.ElementTree as et 
import pandas as pd
import sys
from tqdm import tqdm


def parse_tree(root):
    df_cols = ["ds_name", "id", "frame_num", "orientation", "box_h", "box_w", "box_xc", "box_yc", "appearance", "movement", "role", "context", "situation"]
    df = pd.DataFrame(columns=df_cols)
    d = {}
    ds_name = root.attrib.get('name')
    d["ds_name"] = ds_name
    for frame in tqdm(root):
        frame_num = frame.attrib.get('number')
        d["frame_num"] = frame_num

        objectlist = frame[0]
        # grouplist = frame[1]
        
        for obj in objectlist:
            objectid = obj.attrib.get('id')
            d['id'] = objectid

            orientation = obj[0].text
            d['orientation'] = orientation

            box = obj[1].attrib

            box_h = box.get('h')
            d['box_h'] = box_h

            box_w = box.get('w')
            d['box_w'] = box_w

            box_xc = box.get('xc')
            d['box_xc'] = box_xc

            box_yc = box.get('yc')
            d['box_yc'] = box_yc

            appearance = obj[2].text
            d['appearance'] = appearance

            hypothesis = obj[3][0]
            movement = hypothesis[0].text
            d['movement'] = movement

            role = hypothesis[1].text
            d['role'] = role

            context = hypothesis[2].text
            d['context'] = context

            situation = hypothesis[3].text
            d['situation'] = situation

            df = df.append(d, ignore_index=True)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise EnvironmentError('Not enough arguments specified')
    INFILE = sys.argv[1]
    OUTFILE = sys.argv[2]
    xtree = et.parse("test.xml")
    xroot = xtree.getroot()

    df = parse_tree(xroot)
    df.to_csv(OUTFILE)