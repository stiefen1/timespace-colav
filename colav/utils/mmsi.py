import random

def generate_random_mmsi():
    return random.randint(100000000, 999999999)

def generate_realistic_mmsi():
    # Common country codes (first 3 digits)
    country_codes = [200, 201, 211, 219, 230, 235, 247, 250, 257, 261, 266, 269, 303, 316, 338, 366, 367, 368]
    country_code = random.choice(country_codes)
    
    # Generate remaining 6 digits
    remaining_digits = random.randint(100000, 999999)
    
    return country_code * 1000000 + remaining_digits

def generate_mmsi_formatted():
    # Generate 9-digit number as string to ensure leading zeros if needed
    return int(f"{random.randint(1, 9)}{random.randint(10000000, 99999999)}")



if __name__ == "__main__":
    # Usage
    mmsi = generate_random_mmsi()
    print("random mmsi: ", mmsi)  # e.g., 456789123
    mmsi = generate_realistic_mmsi()
    print("realistic mmsi: ", mmsi)  # e.g., 235123456 (UK vessel)
    mmsi = generate_mmsi_formatted()
    print("formatted mmsi: ", mmsi)
