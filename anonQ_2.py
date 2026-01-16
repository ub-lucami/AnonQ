import pandas as pd
import numpy as np
import warnings

# Constants for input and output file paths
INPUT_FILE_PATH = r'G:\TS\Izvoz_newtimes_month.csv'
OUTPUT_FILE_PATH = 'anonymized_events.csv'
USER_REMOVAL_REPORT_FILE_PATH = 'user_removal_report_counts.csv'
EVENT_REMOVAL_REPORT_FILE_PATH = 'event_removal_report.csv'
# Remove users (True) or events (False)
REMOVE_USERS = False
# specify generalized_event list to be removed before processing
generalized_event_to_remove = [
#    'emergency_button', 
#    'fall_detected', 
#    'fall_cancelled', 
#    'fall_idle', 
    'measurement_event', 
#    'waterleak_detected',
#    'smoke_detected', 
    'power_event', 
    'system_event', 
    ]
# generalized_event_to_remove = [] # (un)comment for quick (no) removal
# Define the contributing columns for anonymized_attribute
contributing_columns = [
    'generalized_event',
#    'event',
#    'year',
#    'month',
#    'day',
#    'hour',
#    'quantized_hour',
    'week_number',
#    'workday_weekend',
    'weekday',   
    'time_period',
]
# Define the column to drop when flattened
flatten_column = 'week_number'
# Set the value of k - level of anonymity
k = 5

# Load your data
df = pd.read_csv(INPUT_FILE_PATH, delimiter=';')

# Extract time components
df['timestamp'] = pd.to_datetime(df['OD_ISO'])

# Parameters for specific quantization/anonymization
hour_block_duration = 3  # Duration of hour blocks
workday_weekend_mapping = {0: 'workday', 1: 'workday', 2: 'workday', 3: 'workday', 4: 'workday', 5: 'weekend', 6: 'weekend'}
# Classification of events
event_mapping = {
    'CONTROL_PANEL.EMERGENCY_BUTTON_PRESSED': 'emergency_button',
    'EMERGENCY_PENDANT.EMERGENCY_BUTTON_PRESSED': 'emergency_button',
    'FALL_DETECTOR.EMERGENCY_BUTTON_PRESSED': 'emergency_button',
    'PULL_CORD.EMERGENCY_BUTTON_PRESSED': 'emergency_button',
    'FALL_DETECTOR.FALL_DETECTION_CANCELED': 'fall_cancelled',
    'FALL_DETECTOR.FALL_DETECTED': 'fall_detected',
    'FALL_DETECTOR.NOT_MOVING': 'fall_idle',
    'BLOOD_PRESSURE_MONITOR.MEASUREMENT_RECEIVED_VALID': 'measurement_event',
    'OXIMETER.MEASUREMENT_RECEIVED_VALID': 'measurement_event',
    'WEIGHT_SCALE.MEASUREMENT_RECEIVED_VALID': 'measurement_event',
    'BLOOD_PRESSURE_MONITOR.BATTERY_EMPTY': 'power_event',
    'BLOOD_PRESSURE_MONITOR.POWER_ON': 'power_event',
    'CONTROL_PANEL.BATTERY_EMPTY': 'power_event',
    'CONTROL_PANEL.BATTERY_FAILURE': 'power_event',
    'CONTROL_PANEL.BATTERY_FAILURE_RESTORED': 'power_event',
    'CONTROL_PANEL.BATTERY_FULL': 'power_event',
    'CONTROL_PANEL.MAINS_POWER_FAILED': 'power_event',
    'CONTROL_PANEL.MAINS_POWER_RESTORED': 'power_event',
    'CONTROL_PANEL.POWER_ON': 'power_event',
    'EMERGENCY_PENDANT.BATTERY_EMPTY': 'power_event',
    'EMERGENCY_PENDANT.BATTERY_FULL': 'power_event',
    'EMERGENCY_PENDANT.POWER_ON': 'power_event',
    'FALL_DETECTOR.BATTERY_EMPTY': 'power_event',
    'FALL_DETECTOR.BATTERY_FAILURE': 'power_event',
    'FALL_DETECTOR.BATTERY_FAILURE_RESTORED': 'power_event',
    'FALL_DETECTOR.BATTERY_FULL': 'power_event',
    'FALL_DETECTOR.POWER_ON': 'power_event',
    'OXIMETER.POWER_ON': 'power_event',
    'PULL_CORD.POWER_ON': 'power_event',
    'SMOKE_DETECTOR.POWER_ON': 'power_event',
    'WEIGHT_SCALE.BATTERY_EMPTY': 'power_event',
    'WEIGHT_SCALE.BATTERY_FULL': 'power_event',
    'SMOKE_DETECTOR.SMOKE_DETECTED': 'smoke_detected',
    'CONTROL_PANEL.MISSING_PROTOCOL_MAPPING': 'system_event',
    'CONTROL_PANEL.SUPERVISORY': 'system_event',
    'CONTROL_PANEL.SUPERVISORY_RESOTRED': 'system_event',
    'EMERGENCY_PENDANT.SUPERVISORY': 'system_event',
    'EMERGENCY_PENDANT.SUPERVISORY_RESOTRED': 'system_event',
    'FALL_DETECTOR.SUPERVISORY': 'system_event',
    'FALL_DETECTOR.SUPERVISORY_RESOTRED': 'system_event',
    'MOTION_DETECTOR.SUPERVISORY': 'system_event',
    'MOTION_DETECTOR.SUPERVISORY_RESOTRED': 'system_event',
    'MOTION_DETECTOR.TAMPER_DETECTED': 'system_event',
    'MOTION_DETECTOR.TAMPER_RESTORED': 'system_event',
    'PULL_CORD.SUPERVISORY': 'system_event',
    'PULL_CORD.SUPERVISORY_RESOTRED': 'system_event',
    'SMOKE_DETECTOR.SMOKE_DETECTION_RESTORED': 'system_event',
    'SMOKE_DETECTOR.SUPERVISORY': 'system_event',
    'SMOKE_DETECTOR.SUPERVISORY_RESOTRED': 'system_event',
    'WATER_LEAK_DETECTOR.SUPERVISORY': 'system_event',
    'WATER_LEAK_DETECTOR.SUPERVISORY_RESOTRED': 'system_event',
    'WATER_LEAK_DETECTOR.WATER_LEAK_RESTORED': 'system_event',
    'WATER_LEAK_DETECTOR.WATER_LEAK_DETECTED': 'waterleak_detected'
}
# Mapping of hours to periods of the day
def map_hour_to_period(hour):
    if 22 <= hour or hour < 6:
        return 'night'
    elif 6 <= hour < 8:
        return 'morning'
    elif 8 <= hour < 17:
        return 'daytime'
    else:
        return 'afternoon'

# cleanup unused columns
df = df.drop(columns=['OD_ISO'])
# count the number of unique users and events in total
total_users_df = df['GUID'].nunique()
total_events_df = len(df)
# Extract conventional datetime components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
# Quantize time
df['quantized_hour'] = (df['hour'] // hour_block_duration) * hour_block_duration
# Extract week number, workday/weekend, weekday, and time period
df['week_number'] = df['timestamp'].dt.isocalendar().week
df['workday_weekend'] = df['timestamp'].dt.weekday.map(workday_weekend_mapping)
df['weekday'] = df['timestamp'].dt.weekday
df['time_period'] = df['hour'].apply(map_hour_to_period)
# Generalize event IDs
df['generalized_event'] = df['dogodek'].map(event_mapping)
# Rename column 'dogodek' to 'event'
df = df.rename(columns={'dogodek': 'event'})

# Drop the original dogodek column if needed
# df = df.drop(columns=['dogodek', 'hour', 'weekday'])

# Function to check k-anonymity
def check_k_anonymity(df, k):
    grouped = df.groupby(['anonymized_attribute']).agg({'GUID': pd.Series.nunique}).reset_index()
    return (grouped['GUID'] >= k).all()

# Function to maximize data retention while achieving k-anonymity by removing violating users
def maximize_data_retention(df, k):
    removed_users = set()
    while not check_k_anonymity(df, k):
        # Find the anonymized attribute causing k-anonymity violation
        grouped = df.groupby(['anonymized_attribute']).agg({'GUID': pd.Series.nunique}).reset_index()
        violating_attribute = grouped[grouped['GUID'] < k].iloc[0]['anonymized_attribute']
        
        # Find the user causing the violation and drop their event
        violating_users = df[df['anonymized_attribute'] == violating_attribute]['GUID'].value_counts()
        user_to_drop = violating_users.idxmin()
        removed_users.add(user_to_drop)
        df = df[df['GUID'] != user_to_drop]
    return df, removed_users

# def maximize_data_retention_remove_events(df, k):
#     i=0
#     while not check_k_anonymity(df, k):
#         i+=1
#         # Find the anonymized attribute causing k-anonymity violation
#         grouped = df.groupby(['anonymized_attribute']).agg({'GUID': pd.Series.nunique}).reset_index()
#         violating_attribute = grouped[grouped['GUID'] < k].iloc[0]['anonymized_attribute']
#         # Find the event causing the violation and drop it
#         violating_events = df[df['anonymized_attribute'] == violating_attribute]
#         event_to_drop = violating_events.index
#         #event_to_drop = violating_events.sample(n=1).index
#         df = df.drop(event_to_drop)
#     print(i)
#     return df, set()

def maximize_data_retention_remove_events(df, k):
    # Count GUIDs per anonymized_attribute
    grouped = df.groupby('anonymized_attribute').agg({'GUID': pd.Series.nunique}).reset_index()   
    # Find all anonymized_attribute with less than k GUIDs
    violating_attributes = grouped[grouped['GUID'] < k]['anonymized_attribute']
    # Delete all occurrences of these violating attributes from df
    df = df[~df['anonymized_attribute'].isin(violating_attributes)]
    return df, set()

# flatten violating column values 
def maximize_data_retention_flatten_events(df, k, flatten):
    if flatten in df.columns:
        # Perform the algorithm
        grouped = df.groupby('anonymized_attribute').agg({'GUID': pd.Series.nunique}).reset_index()
        violating_attributes = grouped[grouped['GUID'] < k]['anonymized_attribute']
        df.loc[df['anonymized_attribute'].isin(violating_attributes), flatten] = 100
        # Count GUIDs per anonymized_attribute
        grouped = df.groupby('anonymized_attribute').agg({'GUID': pd.Series.nunique}).reset_index()   
        # Find all anonymized_attribute with less than k GUIDs
        violating_attributes = grouped[grouped['GUID'] < k]['anonymized_attribute']
        # Delete all occurrences of these violating attributes from df
        df.loc[df['anonymized_attribute'].isin(violating_attributes), flatten] = np.nan
    else:
        print(f"Warning: The column '{flatten}' is not present in the DataFrame. Hierarchical anonymisation will not be applied.")
    return df, set()

# Remove all 'events_to_remove' events
if generalized_event_to_remove:
    df = df[~df['generalized_event'].isin(generalized_event_to_remove)]

# Calculate the total number of unique users before anonymization
total_users_before = df['GUID'].nunique()

# Construct the anonymized_attribute
df['anonymized_attribute'] = df[contributing_columns].astype(str).agg('-'.join, axis=1)
df, removed_users = maximize_data_retention_flatten_events(df, k, flatten_column)
df['anonymized_attribute'] = df[contributing_columns].astype(str).agg('-'.join, axis=1)

if REMOVE_USERS:
    # Maximize data retention while achieving k-anonymity by removing violating users
    df_anonymized, removed_users = maximize_data_retention(df, k)
else:
    # Maximize data retention while achieving k-anonymity by removing violating events
    df_anonymized, removed_users = maximize_data_retention_remove_events(df, k)

# Calculate the total number of unique users after anonymization
total_users_after = df_anonymized['GUID'].nunique()

# Print summary
print(f"{'Metric':<40}{'Value':>10}")
print(f"{'-'*50}")

print(f"{'Number of users (total):':<40}{total_users_df:>10}")
if generalized_event_to_remove:
    print(f"{'Number of users (after event cleanup):':<40}{total_users_before:>10}")
print(f"{'Number of users (anonymized):':<40}{total_users_after:>10}")
print(f"{'Number of events (total):':<40}{total_events_df:>10}")
if generalized_event_to_remove:
    print(f"{'Number of events (after event cleanup):':<40}{len(df):>10}")
print(f"{'Number of events (anonymized):':<40}{len(df_anonymized):>10}")
print(f"{'-'*50}")

# Confirm k-anonymity
if check_k_anonymity(df_anonymized, k):
    print(f"The reduced dataset achieves {k}-anonymity.")
else:
    print(f"The reduced dataset does not achieve {k}-anonymity.")

# Remove unused columns from df_anonymized and save the anonymized data
unused_columns = set(df.columns) - set(contributing_columns) - {'GUID'}
df_anonymized = df_anonymized.drop(columns=unused_columns)
df = df.drop(columns=unused_columns)

# Generate a 1:1 mapping of GUID to new random identifiers
unique_guids = df_anonymized['GUID'].unique()
guid_mapping = {guid: f"ID_{i:05}" for i, guid in enumerate(unique_guids, start=1)}

# Replace GUID with the new random identifiers in the anonymized DataFrame
df_anonymized['GUID'] = df_anonymized['GUID'].map(guid_mapping)

# Optionally, replace GUID in the original DataFrame as well (if needed)
# df['GUID'] = df['GUID'].map(guid_mapping)

df_anonymized.to_csv(OUTPUT_FILE_PATH, index=False, sep=';')

# event counts before and after anonymization (removal), with GUID removed, and save the report
merged_group = pd.merge(
    df.groupby(contributing_columns).agg({'GUID': pd.Series.nunique}).reset_index(),
    df_anonymized.groupby(contributing_columns).agg({'GUID': pd.Series.nunique}).reset_index(),
    on=contributing_columns,
    how='left',
    suffixes=('_original', '_anonymized')
)
#merged_group.to_csv(EVENT_REMOVAL_REPORT_FILE_PATH, index=False, sep=';', float_format='%.0f')

# Add totals row per each unique entry of the first column
unique_entries = merged_group[contributing_columns[0]].unique()
totals_list = []

for entry in unique_entries:
    entry_group = merged_group[merged_group[contributing_columns[0]] == entry]
    total_original = entry_group['GUID_original'].sum()
    total_anonymized = entry_group['GUID_anonymized'].fillna(0).sum()
    totals_list.append({contributing_columns[0]: entry, 'GUID_original': total_original, 'GUID_anonymized': total_anonymized})

totals_df = pd.DataFrame(totals_list)
merged_group = pd.concat([merged_group, totals_df], ignore_index=True)
merged_group.to_csv(EVENT_REMOVAL_REPORT_FILE_PATH, index=False, sep=';', float_format='%.0f')

# Count and save the number of events per each removed user
removed_users_event_counts = df[df['GUID'].isin(removed_users)]['GUID'].value_counts().to_dict()
with open(USER_REMOVAL_REPORT_FILE_PATH, 'w') as f:
    f.write("User;Event Count\n")
    for user, count in removed_users_event_counts.items():
        f.write(f"{user};{count}\n")



