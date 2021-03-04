import sys
import pickle
sys.path.append("..")
import tools
from os.path import join
from tools import process_EFCAMDAT_XML
from collections import Counter
counter = Counter()
learners_by_n  = {}
with open(join(tools.project_folders["EFCAMDAT"],"learners.pickle"),"rb") as learnersf:
    l = pickle.load(learnersf)
    for l_id, l_data in l.items():
        n_writings = len(l_data["writings"])
        counter[n_writings] += 1
        if learners_by_n.get(n_writings):
            learners_by_n[n_writings].append(l_data) 
        else:
            learners_by_n[n_writings]=[l_data] 
with open("learners_by_n.pickle","wb") as learners_by_n_outf, open("learners_count_by_n.pickle","wb") as learners_outf :
    pickle.dump(learners_by_n,learners_by_n_outf)
    pickle.dump(counter, learners_outf)
