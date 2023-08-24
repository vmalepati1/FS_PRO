import pandas as pd
import os
import glob
import re


from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

# print(glob.glob("./data_output/*.csv"))

path = os.path.join(os.getcwd(), "data_output")

# glob.glob(path)

files = glob.glob("./data output/*.csv")

process_ids = []
for f in files:
    if int(re.sub('[^0-9]','', f)) not in process_ids:
        process_ids.append(int(re.sub('[^0-9]','', f))) 


df_decision_fitness = pd.DataFrame()

df_events = pd.DataFrame()

df_components = pd.DataFrame()

for num in process_ids:
    df_decision_fitness = df_decision_fitness.append(pd.read_csv("./data output/decision_fitness %d.csv" %num), ignore_index = True)
    df_events = df_events.append(pd.read_csv("./data output/events %d.csv" %num), ignore_index = True)
    df_components = df_components.append(pd.read_csv("./data output/components %d.csv" %num), ignore_index = True)


df_decision_fitness.to_csv("./processed data/decision_fitness.csv", index = False) 
df_events.to_csv("./processed data/events.csv", index = False) 
df_components.to_csv("./processed data/components.csv", index = False) 