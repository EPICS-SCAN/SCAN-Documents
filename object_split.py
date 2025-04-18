import csv

class ObjectSplit():

    def __init__(self, file_input: str, file_output: str, file_output_adv: str) -> None:
        if not isinstance(file_input, str): 
            raise Exception("File input must be a string")
        if not isinstance(file_output, str):
            raise Exception("File output must be a string")
        if not isinstance(file_output_adv, str):
            raise Exception("File output adv must be a string")
        
        try:
            with open(file_input, 'r') as file:
                pass
                
        except FileNotFoundError:
            raise Exception("Input file not found")
        
        self.file_input = file_input
        self.file_output = file_output
        self.file_output_adv = file_output_adv

    def ds(self):
        open(self.file_output, "w").close()
        with open(self.file_input, 'r') as file:
            content = csv.reader(file)
            prev_distance1 = 0
            prev_distance2 = 0
            prev_distance3 = 0
            curr_distance = 0
            prev_time1 = 0
            prev_time2 = 0
            prev_time3 = 0
            curr_time  = 0
            high = True
            end_count = 0
           
            for lines in content:
                if len(lines) != 2:
                    raise Exception("Input file is not in the format: Time,Distance")
                try:
                    lines[0] = int(lines[0])
                    lines[1] = int(lines[1])
                except ValueError:
                    raise Exception("Input file is not in the format: Time,Distance")
                
                prev_time1 = prev_time2
                prev_time2 = prev_time3
                prev_time3 = curr_time
                curr_time = lines[0]
                prev_distance1 = prev_distance2
                prev_distance2 = prev_distance3
                prev_distance3 = curr_distance
                curr_distance = lines[1]

                if curr_distance < 190 and prev_time1 != 0 and high:
                    with open(self.file_output, "a") as file2:
                        file2.write(f"{prev_time1},{prev_distance1}\n")
                        file2.write(f"{prev_time2},{prev_distance2}\n")
                        file2.write(f"{prev_time3},{prev_distance3}\n")
                        file2.write(f"{curr_time},{curr_distance}\n")
                        high = False
                        end_count = 0
                elif curr_distance < 190 and not high:
                    end_count = 0
                    with open(self.file_output, "a") as file2:
                        file2.write(f"{curr_time},{curr_distance}\n")
                elif end_count == 3:
                    with open(self.file_output, "a") as file2:
                        file2.write("---\n")
                        high = True
                        end_count = 0
                elif curr_distance > 190 and not high:
                    end_count += 1
                    with open(self.file_output, "a") as file2:
                        file2.write(f"{curr_time},{curr_distance}\n")
        
    def ds_adv(self):
        open(self.file_output_adv, "w").close()
        with open(self.file_output, "r") as file:
            content = csv.reader(file)
            high = True
            for lines in content:
                if lines == ["---"]:
                    continue
                if len(lines) != 2:
                    raise Exception("Input file is not in the format: Time,Distance")
                try:
                    lines[0] = int(lines[0])
                    lines[1] = int(lines[1])
                except ValueError:
                    raise Exception("Input file is not in the format: Time,Distance")
            
                if lines[1] < 190:
                    with open(self.file_output_adv, "a") as file2:
                        file2.write(f'{lines[0]},{lines[1]}\n')
                        high = False
                elif not high:
                     with open(self.file_output_adv, "a") as file2:
                        file2.write("---\n")
                        high = True



if __name__ == "__main__":
    test = ObjectSplit("RAK_DATA.TXT", "RAK_DATA_SPLIT.txt", "RAK_DATA_SPLIT_ADV.txt")
    #test.ds()
    test.ds_adv()
    
