import json
from pprint import pprint
from collections import defaultdict

with open('data/rec_proc.json') as json_file:
    models = json.load(json_file)

tags = list(models.keys())

predictions = {tag: {} for tag in tags}

for tag in tags:
    model = models[tag]
    for pos_tag in model['prioritise']:
        predictions[pos_tag][tag] = model["prioritise"][pos_tag]
    for neg_tag in model['deprioritise']:
        predictions[neg_tag][tag] = model["deprioritise"][neg_tag]

pprint(predictions)