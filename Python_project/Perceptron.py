# import numpy
import random
import math


class Perceptron:

    # Sets some instance independent variables
    path_to_training = r"/home/harald/AI-facial_recognition/Assignment_specification/training-A.txt"
    # r"H:\Desktop\AI\AI_2\Test_set\training-A.txt"
    path_to_ans = r"/home/harald/AI-facial_recognition/Assignment_specification/facit-A.txt"
    # r"H:\Desktop\AI\AI_2\Test_set\facit-A.txt"
    path_to_test_images = r"/home/harald/AI-facial_recognition/Assignment_specification/test-B.txt"
    path_to_test_ans = r"/home/harald/AI-facial_recognition/Assignment_specification/facit-B.txt"
    bias = 1 # TODO might want to change bias throughout learning, dunno

    def __init__(self):  # Runs for each instance
        self.epoch = 70 # Add epoch into __init__

        self.temp_variable = self.training_to_dic(self.path_to_training)
        self.training_dic = self.temp_variable[0]  # Returns dictionary and list of keys
        self.training_image_keys = self.temp_variable[1]# Stores the keys belonging to training_set

        self.training_ans = self.training_ans_to_dict(self.path_to_ans) # Same for answers
        self.weight_list = [random.uniform(0.01, 0.2) for x in range(400)]  # TODO might want to modify range

        self.test_images = self.training_to_dic(self.path_to_test_images)[0]
        self.test_ans = self.training_ans_to_dict(self.path_to_test_ans)


    def read_file(self, file_path):
        """
        Reads the file corresponding to the path given
        """
        with open(file_path) as file:
            data_set = file.readlines()
            file.close()
        return data_set

    def training_to_dic(self, path_to_file):
        """
        :return: A finished dictionary from the image format file, key corresponds to list of pixel values
        """
        training_set = self.read_file(path_to_file)  # Load file using read.file()
        list_of_img = []
        list_of_key = []
        for row, line in enumerate(training_set):
            if line.startswith("Image") is True:
                list_of_key.append(line.rstrip())   # Insert image name into keys
                img = []
                for numbers in training_set[row + 1:row + 21]:  # For each row of pixels, format and insert into long list
                    numbers = numbers.strip()
                    numbers = [int(i) / 31 for i in
                               numbers.split()]  # Divide signal by 31 to get fro, 0 to 1
                    img += numbers
                list_of_img.append(img)
        d = {}
        for index, key in enumerate(list_of_key):  # Binds each key (image-name) to list of pixel values
            d[key] = list_of_img[index]
        return d, list_of_key

    def training_ans_to_dict(self, path_to_ans):
        """
        :return: Finnished dictionary with each image name corresponding to right face (1,2,3,4)
        """
        training_ans = self.read_file(path_to_ans)   # Load file using read.file()
        list_of_ans = []
        list_of_key = []
        for line in training_ans:
            if line.startswith("Image") is True:
                list_of_key.append(line.split()[0])
                """ 
                Rows below is for special case, two outcomes instead of four!
                """
                if int(line.split()[1]) == 1 or int(line.split()[1]) == 2:  #TODO Here is where i reduce to two cases
                    list_of_ans.append(1)
                else:
                    list_of_ans.append(0)
        d = {}
        for index, key in enumerate(list_of_key):  # Links the key (image-name) to correct answer (1,2,3,4)
            d[key] = list_of_ans[index]
        return d

    def activation(self, image, set_of_images):  # Takes a randomly generated list of weights and multiplies by signal for all pixel
        """
        Takes the random-generated list of weights and caluculates the activation signal.
        :param image: The image we want to check
        :return: Tanh(X) where x is the activation signal
        """
        activ = Perceptron.bias
        # pixels = self.training_dic[image]
        pixels = set_of_images[image]
        for index, pixel in enumerate(pixels):
            activ += self.weight_list[index] * pixel
        """ Alternative activation functions, check out Leaky RealU- and Softmax-function"""
        return 1.0 if activ >= 1.0 else 0.0
        # return 1 / (1 + math.pow(math.e, -activ))
        #return math.tanh(activ)


    def change_weights(self, error, weights, image):
        """
        Modifies the weight list for each face-recognizing neuron
        :param error: The error between signal vien from activation() and the correct answer
        :param weights: List of weights
        :param image: The image we want to correct for
        :return: New corrected list of weights
        """
        learning_rate = 0.5  # TODO might want to change learning rate
        for index, weight in enumerate(weights):
            weights[index] = weight + (learning_rate * error * self.training_dic[image][index])  # Calculates the
            # weight correction for each weight
        return weights

    def training_of_perceptron(self):
        """
        Trains the weight-list, or the perceptron. Loop that runs activation() and change_weights() for each
        image in the training_set
        :return: Final trained weight_list
        """
        # print(self.training_dic)
        # print(self.weight_list)
        for times in range(self.epoch):
            random.shuffle(self.training_image_keys)
            for key in self.training_image_keys:
                y_ans = self.training_ans[key]
                y_signal = self.activation(key, self.training_dic)
                error = y_ans - y_signal
                self.weight_list= self.change_weights(error, self.weight_list, key)

        print(self.weight_list)
        return self.weight_list  # The trained weight list

    def tester(self):
        accuracy = 0
        for key in self.test_ans:
            predicted = float(self.activation(key, self.test_images))
            expected = float(self.test_ans[key])
            print("Predicted: {}  Expected: {}  Image: {}".format(predicted, expected, key))
            if predicted == expected:
                accuracy += 1
            #print(self.test_images[key])
        print(accuracy/100)
        pass
        """
        Läs in rätt fil och svar
        printa predictred mot ans
        beräkna procenten
        """



p1 = Perceptron()
p1.training_of_perceptron()
p1.tester()

# TODO create a four way network, possibly need to do four different dicts