"""
Maritime Mobile Service Identity (MMSI) utilities.

Provides functions for generating and validating MMSI numbers used
for vessel identification in maritime automatic identification systems (AIS).

Key Functions
-------------
generate_random_mmsi : Generate random 9-digit MMSI
generate_realistic_mmsi : Generate MMSI with realistic country codes
generate_mmsi_formatted : Generate properly formatted MMSI
is_valid_mmsi : Validate MMSI number format
"""

import random

def generate_random_mmsi():
    """
    Generate a random 9-digit MMSI number.
    
    Returns
    -------
    int
        Random MMSI number between 100000000 and 999999999.
        
    Examples
    --------
    >>> mmsi = generate_random_mmsi()
    >>> print(mmsi)  # e.g., 456789123
    """
    return random.randint(100_000_000, 999_999_999)

def generate_realistic_mmsi():
    """
    Generate MMSI with realistic country codes.
    
    Uses common maritime country codes for the first 3 digits,
    making the generated MMSI more realistic for testing.
    
    Returns
    -------
    int
        MMSI number with realistic country code prefix.
        
    Examples
    --------
    >>> mmsi = generate_realistic_mmsi()
    >>> print(mmsi)  # e.g., 235123456 (UK vessel)
    """
    # Common country codes (first 3 digits)
    country_codes = [200, 201, 211, 219, 230, 235, 247, 250, 257, 261, 266, 269, 303, 316, 338, 366, 367, 368]
    country_code = random.choice(country_codes)
    
    # Generate remaining 6 digits
    remaining_digits = random.randint(100_000, 999_999)
    
    return country_code * 1_000_000 + remaining_digits

def generate_mmsi_formatted():
    """
    Generate properly formatted 9-digit MMSI.
    
    Ensures the generated MMSI has exactly 9 digits with
    no leading zeros issue.
    
    Returns
    -------
    int
        Properly formatted 9-digit MMSI number.
        
    Examples
    --------
    >>> mmsi = generate_mmsi_formatted()
    >>> print(len(str(mmsi)))  # Always 9
    """
    # Generate 9-digit number as string to ensure leading zeros if needed
    return int(f"{random.randint(1, 9)}{random.randint(10_000_000, 99_999_999)}")


def is_valid_mmsi(number: int) -> bool:
    """
    Check if a number is a valid MMSI (Maritime Mobile Service Identity).

    Args:
        number (int): The number to check.

    Returns:
        bool: True if the number is a valid MMSI, False otherwise.
    """
    return 100_000_000 <= number <= 999_999_999


if __name__ == "__main__":
    # Usage
    mmsi = generate_random_mmsi()
    print("random mmsi: ", mmsi)  # e.g., 456789123
    mmsi = generate_realistic_mmsi()
    print("realistic mmsi: ", mmsi)  # e.g., 235123456 (UK vessel)
    mmsi = generate_mmsi_formatted()
    print("formatted mmsi: ", mmsi)
