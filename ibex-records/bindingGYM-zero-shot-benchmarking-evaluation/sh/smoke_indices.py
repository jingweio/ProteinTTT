import sys, pandas as pd
ids = pd.read_csv(sys.argv[1])["DMS_id"].tolist()
for a in sys.argv[2:]:
    print(ids.index(a))
