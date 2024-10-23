import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame(): # type: ignore
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Read the dataset from the CSV file
    df = pd.read_csv(r'C:\Users\Shubham\OneDrive\Desktop\Mapup Assesment\MapUp-DA-Assessment-2024\datasets\dataset-2.csv')

    # Create a list of unique IDs
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    
    # Initialize an empty distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)
    
    # Populate the distance matrix with known distances
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Set the distance in the matrix (id_start to id_end and id_end to id_start)
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance  # Ensure symmetry

    # Compute cumulative distances using Floyd-Warshall algorithm
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                # Update distance_matrix[i][j] if a shorter path is found
                if distance_matrix.at[i, k] > 0 and distance_matrix.at[k, j] > 0:
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if new_distance < distance_matrix.at[i, j] or distance_matrix.at[i, j] == 0:
                        distance_matrix.at[i, j] = new_distance

    # Set diagonal to 0 (distance from a point to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix

distance_matrix = calculate_distance_matrix('dataset-2.csv')
print(distance_matrix)



def unroll_distance_matrix(df)->pd.DataFrame(): # type: ignore
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create a list to hold the unrolled data
    unrolled_data = []

    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start and id_end
                distance = distance_matrix.at[id_start, id_end]
                if distance > 0:  # Only include positive distances
                    unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame(): # type: ignore
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter the DataFrame to get distances for the reference_id
    distances_for_reference = df[df["id_start"] == reference_id]

    # Calculate the average distance
    if distances_for_reference.empty:
        return []  # Return empty if no distances found for the reference_id
    
    average_distance = distances_for_reference['distance'].mean()

    # Calculate the 10% threshold
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    # Find all ids within the threshold
    ids_within_threshold = df[
        (df['distance'] >= lower_bound) & 
        (df['distance'] <= upper_bound)
    ]['id_start'].unique()

    # Return a sorted list of ids
    return sorted(ids_within_threshold)

result = find_ids_within_ten_percentage_threshold(distance_matrix, reference_id=1001400)
print(result)



def calculate_toll_rate(df)->pd.DataFrame(): # type: ignore
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
     # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add new columns to the DataFrame
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df

updated_df = calculate_toll_rate(distance_matrix)
print(updated_df)

import pandas as pd
from datetime import time
def calculate_time_based_toll_rates(df)->pd.DataFrame(): # type: ignore
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Add time-related columns with initial values
    df['start_day'] = ''
    df['start_time'] = pd.NaT
    df['end_day'] = ''
    df['end_time'] = pd.NaT

    # Define the time intervals and their corresponding discount factors
    weekday_discount = {
        (time(0, 0), time(10, 0)): 0.8,
        (time(10, 0), time(18, 0)): 1.2,
        (time(18, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount = 0.7

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Populate the DataFrame with time intervals
    row_list = []
    
    for _, row in df.iterrows():
        for day in days_of_week:
            for (start, end), factor in weekday_discount.items():
                row_list.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': start,
                    'end_day': day,
                    'end_time': end,
                    'toll_rate': row['distance'] * factor  # Calculate toll rate for weekdays
                })

            # For weekends (Saturday and Sunday), apply a constant discount factor
            if day in ['Saturday', 'Sunday']:
                row_list.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': time(0, 0),  # Start from midnight
                    'end_day': day,
                    'end_time': time(23, 59, 59),  # End at the last second of the day
                    'toll_rate': row['distance'] * weekend_discount  # Calculate toll rate for weekends
                })
    
    # Create a new DataFrame from the row list
    expanded_df = pd.DataFrame(row_list)

    # Return the expanded DataFrame with the new toll rates
    return expanded_df

distance_df = pd.read_csv('dataset-2.csv')  # Load your DataFrame from a CSV file
updated_df = calculate_time_based_toll_rates(distance_df)
print(updated_df)