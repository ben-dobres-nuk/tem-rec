import json

conversion_dict = {
    "asp.movehomeTRUE": "move_home_asp",
    "asp.homeimpTRUE": "home_imp",
    "asp.holidayTRUE": "holiday",
    "concerned.retireVery concerned": "retire_veryconcerned",
    "concerned.retireSlightly concerned": "retire_slightlyconcerned",
    "concerned.retireNot at all concerned": "retire_notconcerned",
    "concerned.movehomeVery concerned": "movehome_veryconcerned",
    "concerned.movehomeSlightly concerned": "movehome_slightlyconcerned",
    "concerned.movehomeNot at all concerned": "movehome_notconcerned",
    "comp.property.investTRUE": "propinv_completed",
    "financial.knowledgeA lot of knowledge": "knowledge_alot",
    "financial.knowledgeSome knowledge": "knowledge_some",
    "financial.knowledgeVery little knowledge": "knowledge_verylittle",
    "financial.knowledgeNo knowledge": "knowledge_none",
}

with open('data/rec_raw.json') as json_file:
    recs = json.load(json_file)

processed = {}

for main_tag in recs:
    processed[main_tag] = {"prioritise": {}, "deprioritise": {}}

    prioritise_raw = recs[main_tag]["prioritise"]
    deprioritise_raw = recs[main_tag]["deprioritise"]

    def make_priority_dict(priority):

        sub_tag_new = conversion_dict[sub_tag_raw]

        processed[main_tag][priority][sub_tag_new] = recs[main_tag][priority][
            sub_tag_raw]

    for sub_tag_raw in prioritise_raw:

        # populate the tags to be prioritised

        make_priority_dict("prioritise")

    for sub_tag_raw in deprioritise_raw:

        # populate the tags to be deprioritised

        make_priority_dict("deprioritise")

with open('data/rec_proc.json', 'w') as f:
    json.dump(processed, f)
