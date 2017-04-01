import unittest
# from helpers import dataset
import reader

class Test(unittest.TestCase):
    # def test_reader_random(self):
    #     textreader = dataset.TextReader(1, 2)
    #     x, y = textreader.random()
    #     print sum(x)
    #     print y
    def test_Textreader(self):
        textreader = reader.TextReader(data_path = "./data/ptb")
        x, y = textreader.random()
        print sum(x)
        print len(x)
        print y

if __name__ == '__main__':
    unittest.main()
