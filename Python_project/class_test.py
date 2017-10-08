

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

# print(Employee.number_of_employees)
# emp_1 = Employee('Harald', 'Brolin', 200)
# emp_2 = Employee('Hasse', 'Brolin', 200)
# print(Employee.number_of_employees)
# print(emp_2.number_of_employees)

# Decorators

def decorator_function(original_function):
    def wrapper_function():
        print("Wrapper executed befor {}".format(original_function.__name__))
        return original_function()
    return wrapper_function

@decorator_function
def display():
    print("display function ran")

display = decorator_function(display)  # Utan @-decorator så måst evi skriva det här
display()  # Med @ så blir rad 38 == 39

