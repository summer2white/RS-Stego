from steganogan import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import DenseEncoder
from steganogan.loader import DataLoader

# Load the model
steganogan = SteganoGAN.load('dense', cuda=False)

# Encode a message in input.png
steganogan.encode('/root/autodl-tmp/plug-and-play/plug-and-play/experiments/horse_in_mud/samples/0.png', 'output1.png', 'This is a super secret message!')


# Decode the message from output.png
steganogan.decode('output1.png')

