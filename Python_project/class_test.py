

class Employee:
    pay = 400
    number_of_employees = 0

    def __init__(self, fname, lname, pay):  # Körs vid varje instansiering
        self.fname = fname
        self.lname = lname
        self.pay = pay
        Employee.number_of_employees += 1  # Räknas utöver instansieringen, viktigt att nämna klassen och inte self.
        #self.number_of_employees += 1   # Innom instansen men tillhär inte klassen

    def pay_raise(self):
        self.pay = self.pay *1.04

# if __name__ == '__main__':
#     pass

print(Employee.number_of_employees)
emp_1 = Employee('Harald', 'Brolin', 200)
emp_2 = Employee('Hasse', 'Brolin', 200)
print(Employee.number_of_employees)
print(emp_2.number_of_employees)
