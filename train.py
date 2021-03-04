import sys
import pickle
sys.path.append("..")
import tools
from os.path import join
from tools import process_EFCAMDAT_XML

datasets = tools.datasets
writings, meta = process_EFCAMDAT_XML(datasets["EFCAMDAT_raw"])
learners = {}

with open(join(tools.project_folders["EFCAMDAT"],"writings.tsv"),"w") as outf:
    for writing in writings:
        writing_a = writing.attrib
        learner, topic, date, grade, text = [c for c in writing]
        learner_a = learner.attrib
        writing_data = {
            "id": writing_a["id"],
            "level": writing_a["level"],
            "unit": writing_a["unit"],
            "topic_id": topic.attrib["id"],
            "topic_name": topic.text,
            "grade": grade.text,
            "text": text.text
        }
        if learners.get(learner_a["id"],False):
            learners[l["id"]]["writings"].append(writing_data)
        else:
            learner_data = {
                    "id": learner_a["id"], 
                    "nationality": learner_a["nationality"], 
                    "writings": [writing_data]
            }
            learners[learner_data["id"]] = learner_data
        l = learner_data
        w = writing_data
        writing_str_data = f'{w["id"]}\t{l["id"]}\t{l["nationality"]}\t{w["level"]}\t{w["unit"]}\t{w["topic_id"]}\t{w["topic_name"]}\t{w["grade"]}\t{w["text"]}'.replace("\n","")
        outf.write(writing_str_data)
        outf.write("\n")

with open(join(tools.project_folders["EFCAMDAT"],"learners.pickle"),"wb") as learnersf:
    pickle.dump(learners, learnersf)
