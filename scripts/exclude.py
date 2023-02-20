import csv
import json
from collections import defaultdict

included_subjects = {}
all_subject_scores = {}
count_scores = defaultdict(int)
with open('Copy of CBfilters_120922 - Sheet1.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        # print(row)
        score = int(row[-1])
        count_scores[score] += 1
        all_subject_scores[row[1]] = score
        if score < 4:
            included_subjects[row[1]] = score

print(len(all_subject_scores))
print(len(included_subjects))
print(count_scores)

with open('subject_filterscores.json', 'w') as f:
    json.dump(all_subject_scores, f)

# with open('included_subjects.json', 'w') as f:
#     json.dump(included_subjects, f)
