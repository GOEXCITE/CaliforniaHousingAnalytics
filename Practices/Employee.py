#!/usr/bin/python3

class Employee:
    # name = ""
    # salary = 0
    constant_var = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def __str__(self):
        str(self)

    def rename(self, nName):
        self.name = nName

    def print(self):
        print(self.name)
        print(self.salary)
        print(self.constant_var)

sample = Employee(name="tiantian", salary=500)
print(sample.name)
print(sample.salary)
print(sample.constant_var)


