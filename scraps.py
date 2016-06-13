import codecs
from keras.utils.data_utils import get_file


nietzsche_data = get_file('nietzsche.txt', 
                          origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = codecs.open(nietzsche_data, encoding='utf-8').read().lower()
