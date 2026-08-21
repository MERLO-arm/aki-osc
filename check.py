import pandas as pd
df = pd.read_parquet('data/fulfulde_run/train.parquet')
samples = df.sample(10, random_state=42)[['audio_filepath', 'duration', 'text_clean']].to_dict('records')
for i, s in enumerate(samples, 1):
    path = s['audio_filepath']
    dur = s['duration']
    text = s['text_clean']
    print(f'{i}. **Audio** : [{path}](file:///Users/ekwali/multilingual_asr_pipeline/{path})')
    print(f'   - **Durée** : {dur}s')
    print(f'   - **Texte** : `{text}`\n')
