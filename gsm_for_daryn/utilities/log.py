

class log:
    def __init__(self,f,name = "This is a logfile"):
        self.FNAME = f

        F = open(self.FNAME,"w")
        F.write(name + "\n")
        F.close()

    def log(self,data,PRINT = True):

        if PRINT:
            print(data)

        F = open(self.FNAME,"a")
        F.write(str(data) + "\n")
        F.close()

