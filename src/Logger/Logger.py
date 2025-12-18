import csv

class Logger:
    def __init__(self, path):
        self.f = open(path, "w", newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["epoch", "loss", "acc"])

    def log(self, epoch, loss, acc):
        self.writer.writerow([epoch, loss, acc])

    def close(self):
        self.f.close()