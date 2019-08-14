import json

conversion_dict = {
    "asp.movehomeTRUE": "LifeContent_Myhome_Plantobuy",
    "asp.homeimpTRUE": "ProductContent_Mortgages_HomeImprovement",
    "asp.holidayTRUE": "LifeContent: Luxury",
    "concerned.retireVery concerned": "LifeContent_LaterLife_LivingWell_VeryConcerned",
    "concerned.retireSlightly concerned": "life_laterlife_livingwell.low",
    "concerned.retireNot at all concerned": "life_laterlife_livingwell.none",
    "concerned.movehomeVery concerned": "LifeContent_MyHome_WorriedAboutBuying_VeryConcerned",
    "concerned.movehomeSlightly concerned": "life_myhome_worrybuy.low",
    "concerned.movehomeNot at all concerned": "life_myhome_worrybuy.none",
    "comp.property.investTRUE": "LifeContent_ MyHome_PropertyInvestment",
    "financial.knowledgeA lot of knowledge": "knowledge.none",
    "financial.knowledgeSome knowledge": "knowledge.some",
    "financial.knowledgeVery little knowledge": "knowledge.verylow",
    "financial.knowledgeNo knowledge": "knowledge.none",
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

        # remove negative tags from our set

        if not any(substring in sub_tag_new
                   for substring in ["low", "none", "verylow"]):

            processed[main_tag][priority][sub_tag_new] = recs[main_tag][
                priority][sub_tag_raw]

    for sub_tag_raw in prioritise_raw:

        # populate the tags to be prioritised

        make_priority_dict("prioritise")

    for sub_tag_raw in deprioritise_raw:

        # populate the tags to be deprioritised

        make_priority_dict("deprioritise")

with open('data/rec_proc.json', 'w') as f:
    json.dump(processed, f)
