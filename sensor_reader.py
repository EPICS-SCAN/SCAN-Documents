import sys
import numpy as np

class SensorDataReader:
    """
    Reads ultrasonic sensor data from a file and extracts objects separated by "---".
    Data format expected:
        time_ms,distance_cm
    Both fields should be integers.
    """

    def __init__(self, filename):
        self.filename = filename
        self.objects = []
        self._parse_file()

    def _parse_file(self):
        """Parse file and split into objects."""
        with open(self.filename, "r") as f:
            current_object = []

            for line in f:
                line = line.strip()

                # delimiter
                if line == "---":
                    if current_object:
                        self.objects.append(np.array(current_object, dtype=int))
                        current_object = []
                    continue

                # skip empty lines or comments
                if not line or line.startswith("#"):
                    continue

                # parse "int,int"
                parts = line.split(",")
                if len(parts) != 2:
                    continue

                try:
                    t = int(parts[0])
                    d = int(parts[1])
                    current_object.append([t, d])
                except ValueError:
                    continue

            # append last object
            if current_object:
                self.objects.append(np.array(current_object, dtype=int))

    def get_object_count(self):
        return len(self.objects)

    def get_object(self, index):
        if 0 <= index < len(self.objects):
            return self.objects[index]
        else:
            raise IndexError(f"Object index {index} out of range")


# Command-line interface for C
def print_usage():
    print("Usage:")
    print("  python sensor_reader.py <file> count")
    print("  python sensor_reader.py <file> <object_index>")


if __name__ == "__main__":

    if len(sys.argv) >= 3:
        datafile = sys.argv[1]
        cmd = sys.argv[2]

        reader = SensorDataReader(datafile)

        # Case 1: return object count
        if cmd == "count":
            print(reader.get_object_count())
            sys.exit(0)

        # Case 2: return specific object
        try:
            index = int(cmd)
            obj = reader.get_object(index)

            # First print number of points
            print(len(obj))

            # Then print each point as "int,int"
            for t, d in obj:
                print(f"{t},{d}")

            sys.exit(0)

        except Exception:
            print_usage()
            sys.exit(1)

    # No arguments → run in normal human-readable mode
    print("\nPython reader expected arguments but none were provided.\n")
    print_usage()
