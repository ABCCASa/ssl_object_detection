class Person:
    def __init__(self):
        self.x = 1
        self.y = "hi"
        print("hi")
        print(self.__dict__)


p = Person()

