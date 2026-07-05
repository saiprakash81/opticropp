import numpy as np
import pandas as pd
import os

def generate_crop_dataset(n_per_crop=100, seed=42):
    np.random.seed(seed)

    crops = {
        'rice':        dict(N=(60,100), P=(40,60),  K=(40,50),  temp=(20,27), hum=(80,90), pH=(5.5,7.0), rain=(200,300)),
        'maize':       dict(N=(60,80),  P=(50,65),  K=(18,25),  temp=(18,27), hum=(55,70), pH=(5.5,7.0), rain=(60,120)),
        'chickpea':    dict(N=(40,60),  P=(60,80),  K=(75,95),  temp=(15,23), hum=(14,25), pH=(6.0,9.0), rain=(60,100)),
        'kidneybeans': dict(N=(20,40),  P=(60,80),  K=(18,30),  temp=(18,25), hum=(18,25), pH=(5.5,7.0), rain=(100,150)),
        'pigeonpeas':  dict(N=(20,40),  P=(65,85),  K=(18,30),  temp=(18,25), hum=(40,50), pH=(5.0,7.0), rain=(100,200)),
        'mothbeans':   dict(N=(20,40),  P=(40,60),  K=(18,30),  temp=(25,35), hum=(50,70), pH=(3.5,7.0), rain=(50,150)),
        'mungbean':    dict(N=(20,40),  P=(40,60),  K=(18,30),  temp=(25,35), hum=(85,95), pH=(6.0,7.0), rain=(40,100)),
        'blackgram':   dict(N=(30,50),  P=(60,80),  K=(16,25),  temp=(28,36), hum=(65,70), pH=(6.0,7.0), rain=(65,100)),
        'lentil':      dict(N=(18,25),  P=(60,80),  K=(18,25),  temp=(16,24), hum=(64,70), pH=(6.0,8.0), rain=(40,50)),
        'pomegranate': dict(N=(18,25),  P=(16,25),  K=(38,48),  temp=(20,27), hum=(88,95), pH=(5.5,7.0), rain=(100,200)),
        'banana':      dict(N=(100,120),P=(72,84),  K=(48,55),  temp=(25,30), hum=(75,90), pH=(5.5,7.0), rain=(100,200)),
        'mango':       dict(N=(18,25),  P=(15,25),  K=(29,36),  temp=(27,35), hum=(50,60), pH=(4.0,7.0), rain=(100,200)),
        'grapes':      dict(N=(18,23),  P=(122,134),K=(198,212),temp=(8,17),  hum=(81,88), pH=(5.5,7.0), rain=(55,70)),
        'watermelon':  dict(N=(99,104), P=(16,22),  K=(48,55),  temp=(24,27), hum=(85,92), pH=(6.0,7.0), rain=(40,60)),
        'muskmelon':   dict(N=(99,104), P=(16,22),  K=(48,55),  temp=(28,32), hum=(88,96), pH=(6.0,7.0), rain=(20,30)),
        'apple':       dict(N=(20,25),  P=(125,135),K=(198,212),temp=(20,24), hum=(90,96), pH=(5.5,7.0), rain=(110,120)),
        'orange':      dict(N=(18,23),  P=(15,22),  K=(10,16),  temp=(10,15), hum=(90,96), pH=(6.0,7.5), rain=(110,120)),
        'papaya':      dict(N=(48,55),  P=(58,66),  K=(48,55),  temp=(33,38), hum=(90,97), pH=(6.0,7.0), rain=(140,160)),
        'coconut':     dict(N=(19,23),  P=(14,18),  K=(29,36),  temp=(26,28), hum=(89,96), pH=(5.0,8.0), rain=(140,160)),
        'cotton':      dict(N=(117,122),P=(45,52),  K=(42,48),  temp=(23,27), hum=(77,85), pH=(6.0,8.0), rain=(65,80)),
        'jute':        dict(N=(77,82),  P=(45,52),  K=(38,45),  temp=(24,27), hum=(78,85), pH=(6.0,7.0), rain=(170,190)),
        'coffee':      dict(N=(99,104), P=(27,34),  K=(28,35),  temp=(25,28), hum=(58,65), pH=(6.0,7.0), rain=(150,170)),
    }

    rows = []
    for crop, params in crops.items():
        for _ in range(n_per_crop):
            row = {
                'N':           round(np.random.uniform(*params['N']), 2),
                'P':           round(np.random.uniform(*params['P']), 2),
                'K':           round(np.random.uniform(*params['K']), 2),
                'temperature': round(np.random.uniform(*params['temp']), 2),
                'humidity':    round(np.random.uniform(*params['hum']), 2),
                'ph':          round(np.random.uniform(*params['pH']), 2),
                'rainfall':    round(np.random.uniform(*params['rain']), 2),
                'label':       crop,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Crop_recommendation.csv')
    df = generate_crop_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} — {len(df)} rows, {df['label'].nunique()} crops")
