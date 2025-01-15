import json
from collections import Counter

import pandas as pd

# this dictionary is generated manually from the results of the categorization experiments to reduce redundant and repeated classes by the LLM
MERGED_CLASS_LABELS = {
'person': ['female', 'person', 'group of people', 'group', 'male', 'family', 'team'],
'location': ['location or place', 'castle', 'city', 'locations or places', 'location or places', 'location', 'indoor scene/outdoor scene'],
'action': ['actions', 'action'],
'food & drink': ['food', 'drink', 'taste', 'dessert', 'flavor', 'fruits', 'condiment','meal','sauce', 'cuisine','fruit'],
'objects': ['toy', 'object', 'toys', 'tools', 'objects', 'tool',  'container'],
'UNKNOWN': ['', 'UNKNOWN','purpose', 'atmosphere', 'ornate'],
'animal': ['animal'],
'sport': ['sports', 'sport'],
'outdoor scene': ['outdoor scene','street', 'trail', 'graffiti','store','building', 'road','path', 'buildings','background'],
'indoor scene': ['floor', 'furniture', 'window', 'wall', 'light', 'rooms', 'indoor scene','house','door', 'doorway', 'room', 'windows', 'lighting'],
'culture': ['culture', 'art','architecture', 'art style'],
'color': ['color'],
'clothing': ['accessories', 'jewelry', 'clothing'],
'quantity & number & size': ['number', 'numbers', 'miniature', 'quantity','size','weight'],
'plants & flowers': ['flower', 'plant', 'plants'],
'nature': ['enclosure', 'waterfall', 'nature', 'water', 'body of water', 'mountain', 'fire'],
'events':['holiday', 'event', 'award', 'holiday event'],
'time':['season','time', 'month'],
'weather':['weather', 'temperature'],
'material & texture': ['material','wooden','wood', 'texture', 'metal', 'surface'],
'shape & style & state': ['shape', 'style', 'pattern', 'state', 'structure','liquid'],
'traffic':['speed', 'sign', 'signs', 'transportation', 'direction', 'vehicle', 'part of vehicle'],
'human attributes': ['hair style','age', 'young', 'facial expression', 'hair color', 'emotion', 'old', 'body part', 'language'],
'position':['position', 'orientation', 'relation', 'relations'],
'interaction':['interaction', 'interactions'],
'technology':['technology', 'electronics'],
'preposition':['preposition'],
'movie & game':['theme', 'movie', 'sound','title','game'],
'story':['mythical creature','fairy'],
'company':['company','company name', 'brand','service'],
'composition':['composition'],
'nationality':['nationality'],
'money':['money'],
'text':['text'],
'comparison':['comparison']
}

def handle_duplicates(ordered_pairs):
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            if isinstance(d[k], list):
                d[k].append(v)
            else:
                d[k] = [d[k], v]
        else:
            d[k] = [v]
    return d

#nested_dict = {'a': {'b': {'c': 'value'}}}
def flatten_dict(nested_dict):
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value))
        else:
            flat_dict[key] = value
    return flat_dict

def merg_list(my_list):
    result = []

    for item in my_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)

    return result

def get_all_defined_classes_per_layer(data, layer_no, batch_size):
    curr_layer_class_dist = []
    exceptions = []
    for i in range(batch_size):
        try:
            my_dict = json.loads(data[i], object_pairs_hook=handle_duplicates) #ast.literal_eval(data[f'layer_{layer_no}'][i])
            top3_classes = merg_list(list(flatten_dict(my_dict).values()))
            #print(top3_classes)
            classes = [c[0] for c in top3_classes if isinstance(c, list)]
            curr_layer_class_dist.append(classes)
        except (SyntaxError, ValueError):
            exceptions.append((i, layer_no))
    return curr_layer_class_dist, layer_no, exceptions

def get_frequency(lst, final_list_classes):
    # Count the frequency of each string
    freq = Counter(lst)
    
    # Calculate the total length of the list
    #total = len(lst)
    #total = len(final_list_classes)
    # Sort the list of tuples by frequency in descending order
    sorted_lst = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate the percentage of each string
    #result = [(string, (count / total) * 100) for string, count in sorted_lst]

    per_layer_caption_dist={}

    for c in final_list_classes:
        per_layer_caption_dist[c] = 0
    for string, count in sorted_lst:
        per_layer_caption_dist[string] += count

    return per_layer_caption_dist

def calculate_percentages(caption):
    total = sum(caption.values())
    return {k: v*100 / total for k, v in caption.items()}

def get_layer_class_dist(curr_layer_class_dist, final_list_classes):
    
    all_samples_per_layer = []
    for lst in curr_layer_class_dist:
        all_samples_per_layer.append(get_frequency(lst, final_list_classes)) 
    # Calculate percentages for each caption
    percentages = [calculate_percentages(caption) for caption in all_samples_per_layer]

    # Aggregate percentages across all captions
    aggregated_percentages = {k: sum(d[k] for d in percentages)/len(curr_layer_class_dist) for k in percentages[0]}
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(aggregated_percentages.items()), columns=['Class', 'Percentage'])
    #df = df.set_index('Class')
    return df

def get_adapted_curr_layer_class_dist(top1,layer_no,all_labels):
    count = []
    adapted_layer = []
    for cap in top1[f'layer_{layer_no}']:
        adapted_cap = []
        for c in cap:
            if c in list(all_labels.keys()):
                adapted_cap.append(all_labels[c])
            else:
                count.append(c)
                adapted_cap.append('UNKNOWN') # as it is not clear what these exceptions are, they are replaced by UNKNOWN flag
        adapted_layer.append(adapted_cap)

    return adapted_layer, count

def remove_empty_lists(list_of_lists):
    # Using list comprehension to filter out empty lists
    return [lst for lst in list_of_lists if lst]


def map_all_classes_to_merged_list():
    list_classes = list(MERGED_CLASS_LABELS.keys())
    # flip the keys and values for mapping
    all_labels = {}
    for key,value in MERGED_CLASS_LABELS.items():
        
        for v in value:
            all_labels[v] = key

    return all_labels, list_classes