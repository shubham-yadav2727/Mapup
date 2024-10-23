from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    
    for i in range(0, length, n):
        group = []
        for j in range(min(n, length - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    return result

lst = [1, 2, 3, 4, 5, 6, 7, 8]
n = 3
output = reverse_by_n_elements(lst, n)
print(output)


from collections import defaultdict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = defaultdict(list)
    
    for string in lst:
        length_dict[len(string)].append(string)
    return dict(sorted(length_dict.items()))

lst = ["apple", "bat", "car", "elephant", "dog", "bear"]
output = group_by_length(lst)
print(output)


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten dictionaries
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                # Handle lists by iterating over elements and using index as key part
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        # If the list element is a dictionary, recursively flatten
                        items.extend(_flatten(item, list_key).items())
                    else:
                        # Otherwise, append the list element
                        items.append((list_key, item))
            else:
                # If it's neither dict nor list, add the item to the result
                items.append((new_key, v))
        return dict(items)
    
    # Call the recursive helper function
    return _flatten(nested_dict)

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

output = flatten_dict(nested_dict)
print(output)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, used):
        # If the current path is a complete permutation, add it to the result
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            # Skip if the number is already used in this permutation
            if used[i]:
                continue
            # Skip duplicates: if nums[i] is the same as nums[i-1] and the previous was not used
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            
            # Mark the number as used and add it to the current path
            used[i] = True
            path.append(nums[i])
            
            # Recurse with the updated path and used array
            backtrack(path, used)
            
            # Backtrack by removing the number from the current path and marking it as unused
            path.pop()
            used[i] = False

    # Sort the numbers to handle duplicates efficiently
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    # Start backtracking
    backtrack([], used)
    
    return result

input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define regular expressions for the three date formats
    date_pattern = r'\b(\d{2}-\d{2}-\d{4})\b'  # dd-mm-yyyy
    date_pattern += r'|\b(\d{2}/\d{2}/\d{4})\b'  # mm/dd/yyyy
    date_pattern += r'|\b(\d{4}\.\d{2}\.\d{2})\b'  # yyyy.mm.dd
    
    # Use re.findall to get all matches in the text
    matches = re.findall(date_pattern, text)
    
    # Flatten the result and filter out empty strings (since re.findall returns a tuple per match)
    return [date for match in matches for date in match if date]


text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)


import polyline
from math import radians, cos, sin, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
    
    Returns:
        float: Distance between the two points in meters.
    """
    R = 6371000  # Radius of Earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string to get the coordinates
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame with latitude and longitude
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Add a column for distance, initially set to 0
    df['distance'] = 0.0
    
    # Calculate distance for each successive row using the Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df


polyline_str = 'gfo}EtohhU_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)




def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    if not matrix or not matrix[0]:
        return []  # Return an empty list if the matrix is empty

    n = len(matrix)
    m = len(matrix[0])  # Assuming all rows have the same number of columns

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(m)]  # New dimensions after rotation
    for i in range(n):
        for j in range(m):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Multiply each element by the sum of its original row and column indices
    transformed_matrix = [[0] * m for _ in range(n)]  # Initialize transformed matrix
    for i in range(n):
        for j in range(m):
            original_row_index = i
            original_col_index = j
            index_sum = original_row_index + original_col_index
            transformed_matrix[i][j] = rotated_matrix[i][j] * index_sum

    return transformed_matrix


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)



def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Combine start and end dates and times into a single datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Set multi-index using 'id' and 'id_2'
    df.set_index(['id', 'id_2'], inplace=True)
    
    # Group by multi-index
    grouped = df.groupby(level=['id', 'id_2'])
    
    # Function to check completeness for each group
    def completeness_check(group):
        # Create a date range for the week
        days_of_week = pd.date_range(start=group['start_datetime'].min().floor('D'),
                                      end=group['end_datetime'].max().floor('D'),
                                      freq='D')
        
        for day in days_of_week:
            day_start = day.replace(hour=0, minute=0, second=0)
            day_end = day.replace(hour=23, minute=59, second=59)
            
            # Check if there's at least one start and one end within the day's range
            if not ((group['start_datetime'].between(day_start, day_end).any() and 
                     group['end_datetime'].between(day_start, day_end).any())):
                return True  # Incomplete coverage for this pair
        
        return False  # Complete coverage for this pair

    # Apply completeness check and return as boolean series
    result = grouped.apply(completeness_check)
    
    return result

# df = pd.read_csv('dataset-1.csv')
# incorrect_timestamps = time_check(df)
# print(incorrect_timestamps)
