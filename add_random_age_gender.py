import sys
import pandas as pd
import random

GENDERS = ['Male', 'Female']
AGES = ['0-18', '19-25', '26-40', '41-60', '60+']


def add_random_age_gender(df):
    genders = {}
    ages = {}
    for i in df['id'].unique():
        genders[i] = random.choice(GENDERS)
        ages[i] = random.choice(AGES)

    df['gender'] = df['id'].apply(lambda x: genders[x])
    df['age'] = df['id'].apply(lambda x: ages[x])

    return df


import pandas as pd
if __name__ == "__main__":
    IN_FNAME = sys.argv[1]
    OUT_FNAME = sys.argv[2]
    
    df = pd.read_csv(IN_FNAME, index_col=0)
    
    df = add_random_age_gender(df)

    df.to_csv(OUT_FNAME)