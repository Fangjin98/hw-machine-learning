from utils.constants import attribute_values_map

def map_value(data):
    res=[]
    if(len(data)>1):
        res.extend(attribute_values_map['buying'][data[0]])
        res.extend(attribute_values_map['maint'][data[1]])
        res.extend(attribute_values_map['doors'][data[2]])
        res.extend(attribute_values_map['persons'][data[3]])
        res.extend(attribute_values_map['lug_boot'][data[4]])
        res.extend(attribute_values_map['safety'][data[5]])
    else:
        res=attribute_values_map['labels'][data [0]]

    return res

def formatting(data):
    if(data[0]>0.5): data[0]=1
    else: data[0]=0

    if(data[1]>0.5): data[1]=1
    else: data[1]=0

    return data
