def load_data(file_path):
    import pandas as pd
    
    # Load the IMDB dataset from the specified path
    df = pd.read_csv(file_path)
    
    # Convert sentiment to binary labels
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    return df