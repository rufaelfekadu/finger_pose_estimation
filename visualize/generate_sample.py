import random

def generate_realistic_sample_angles(num_frames, num_key_points):
    with open('realistic_sample_angles.txt', 'w') as file:
        for _ in range(num_frames):
            # Initialize angles with random values
            angles = [random.uniform(0, 360) for _ in range(num_key_points)]

            # Apply constraints or patterns for more realistic angles
            for i in range(num_key_points):
                # Limit angles to a realistic range (e.g., 0 to 180 degrees)
                angles[i] = max(0, min(180, angles[i]))

                # Introduce some correlation between neighboring joints
                if i > 0:
                    angles[i] = min(angles[i], angles[i - 1] + 30)

            line = ' '.join(map(str, angles))
            file.write(line + '\n')

if __name__ == "__main__":
    num_frames = 100  # Adjust the number of frames as needed
    num_key_points = 21

    generate_realistic_sample_angles(num_frames, num_key_points)

    print("Realistic sample angles generated and saved to 'realistic_sample_angles.txt'.")
