from dataset_generator import generate_data, load_data
import string

def main():
    generate_data(base_size=100000, base_length=10, char_n=10, filepath='data2', fn = 'first')
main()