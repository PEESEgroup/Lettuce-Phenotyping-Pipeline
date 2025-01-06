import pandas as pd

# Load and preprocess the height dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Calculate the day number from the timestamp
    data['day'] = (data.index - data.index[0]).days

    # Define growth stages
    def assign_stage(day):
        if day <= 4:
            return 'Seedling'  # Days 0-4
        elif day <= 9:
            return 'Cupping'   # Days 5-9
        elif day <= 13:
            return 'Rosette'   # Days 10-13
        else:
            return 'Heading'   # Days 14-18

    data['stage'] = data['day'].apply(assign_stage)
    return data

# Define the REX and ROUXAI sample columns
def define_sample_columns():
    rex_height_samples = [f'height_{i}' for i in range(0, 20)]   # height_0 to height_19
    rouxai_height_samples = [f'height_{i}' for i in range(20, 40)]   # height_20 to height_39
    return rex_height_samples, rouxai_height_samples
