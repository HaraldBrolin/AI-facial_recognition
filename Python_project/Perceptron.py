# import numpy
import random
import math


class Perceptron:
    path_to_training = r"/home/harald/AI-facial_recognition/Assignment_specification/training-A.txt"
    # r"H:\Desktop\AI\AI_2\Test_set\training-A.txt"
    path_to_ans = r"/home/harald/AI-facial_recognition/Assignment_specification/facit-A.txt"
    # r"H:\Desktop\AI\AI_2\Test_set\facit-A.txt"
    bias = 1

    def __init__(self):
        self.training_set = self.training_to_dic()
        self.training_ans = self.training_ans_to_dict()
        self.weight_list = [random.uniform(0.01, 0.2) for x in range(400)]

    def read_file(self, file_path):
        with open(file_path) as file:
            data_set = file.readlines()
            file.close()
        return data_set

    def training_to_dic(self):
        training_set = self.read_file(Perceptron.path_to_training)
        list_of_img = []
        list_of_key = []
        for row, line in enumerate(training_set):
            if line.startswith("Image") is True:
                list_of_key.append(line.rstrip())
                img = []
                for numbers in training_set[row + 1:row + 21]:
                    numbers = numbers.strip()
                    # numbers = [int(i)/31 for i in numbers.split()]
                    numbers = [int(i) / 31 for i in
                               numbers.split()]  # Signal goes from -1 to 1, this is since every function given needs -1 to 1
                    img += numbers
                list_of_img.append(img)
        d = {}
        for index, key in enumerate(list_of_key):
            d[key] = list_of_img[index]
        return d

    def training_ans_to_dict(self):
        training_ans = self.read_file(Perceptron.path_to_ans)
        list_of_ans = []
        list_of_key = []
        for line in training_ans:
            if line.startswith("Image") is True:
                list_of_key.append(line.split()[0])
                if int(line.split()[1]) == 1 or int(line.split()[1]) == 2:
                    list_of_ans.append(1)
                else:
                    list_of_ans.append(0)
                    # list_of_ans.append(line.split()[1])
        d = {}
        for index, key in enumerate(list_of_key):
            d[key] = list_of_ans[index]
        return d

    def activation(self, image):  # Takes a randomly generated list of weights and multiplies by signal for all pixel
        activ = Perceptron.bias
        pixels = self.training_set[image]
        for index, pixel in enumerate(pixels):
            activ += self.weight_list[index] * pixel
        # return 1.0 if activ >= 1.0 else 0.0
        # return 1 / (1 + math.pow(math.e, -activ))
        return math.tanh(activ)


    def change_weights(self, error, weights, image):
        learning_rate = 0.5
        for index, weight in enumerate(weights):
            weights[index] = weight + (learning_rate * error * self.training_set[image][index])
        return weights

    def training_of_perceptron(self):
        print(self.training_set)
        # self.training_ans
        # self.weight_list
        print(self.weight_list)
        for image in self.training_set:
            y_ans = self.training_ans[image]
            y_signal = self.activation(image)
            # print(y_ans, y_signal)
            error = y_ans - y_signal
            # error = self.training_ans[image] - self.activation(image)
            self.weight_list= self.change_weights(error, self.weight_list, image)
            #print(self.weight_list)
        return self.weight_list  # The trained weight list



p1 = Perceptron()
p1.training_of_perceptron()

# p1_ans = p1.training_ans_to_dict()
# p1_training = p1.training_to_dic()

#
# for image in p1_training:
#     print(p1.activation(image))


# p1_activation = p1.activation(p1_training)
# weights = [0 for x in range(401)]  # Create list of random numbers from 0 to 1
# answer = training_ans_to_dict(read_file(r"H:\Desktop\AI\AI_2\Test_set\facit-A.txt"))
# training_set = training_to_dic(read_file(r"H:\Desktop\AI\AI_2\Test_set\training-A.txt"))
#
# accuracy = []
#
# for index, tuple in enumerate(answer.items()):
#     correct = tuple[1]
#     prediction = activation(training_set[tuple[0]], weights)
#     #print(prediction)
#     error = correct - prediction
#     weights = change_weights(error, weights, training_set[tuple[0]])
#     if prediction >= 0.5:
#         print(correct - 1)
#     else:
#         print(correct)
#     #print(error)
#
#

# activ = activation(training_set["Image1"], weights)
# error = answer["Image1"] - activ
#
# change_weights(error, weights, training_set["Image1"])




