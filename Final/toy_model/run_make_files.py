from toy_firstchar_model import ToyFirstCharModel
import string

def main():
    tf = ToyFirstCharModel()
    tf.generate_data(base_size = 1000000, base_length=60, valid_chars=string.digits)
    print("File generation complete!")
main()