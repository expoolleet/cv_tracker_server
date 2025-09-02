from pathlib import Path
import csv
import datetime

class DataHandler:
    def __init__(self, path=None, file=None, folder=None):
        folder_name = folder if folder else "data"
        base_path = Path(path).resolve() / folder_name if path else Path(__file__).resolve().parent / folder_name
        Path(base_path).mkdir(parents=True, exist_ok=True)
        file_name = file if file else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = base_path / f"{file_name}"
        
    def save(self, data):
        pass
            
    def read(self):
        pass  
    
class CSVHandler(DataHandler):   
    def __init__(self, path=None, file=None, folder=None):
        super().__init__(path, file, folder)
        self.file_path = Path(str(self.file_path) + ".csv")
        print(f"DataHandler is created for {self.file_path}")
    
    def save(self, data):
        with open(self.file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(data["fields"])
            writer.writerows(data["rows"])
            
    def read(self):
        rows = []
        with open(self.file_path, "r") as f:
            reader = csv.reader(f)
            fields = next(f)
            for row in reader:
                rows.append(row)
        return fields, rows