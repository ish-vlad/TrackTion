import sys
import pandas as pd
from sqlalchemy import create_engine

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise OSError('Not enough arguments specified')
    
    FNAME = sys.argv[1]
    TABLENAME = sys.argv[2]

    df = pd.read_csv(FNAME, index_col=0)

    engine = create_engine('postgresql://myuser:tracktion@localhost:5432/tracktion')

    df.to_sql(TABLENAME, engine)