# Sound Processing project

This is a written project with regard to processing sound using keras deep learning library. These are the planned application:
- [x] Genres classification
- [ ] Music auto-tagging
- [ ] Instrumental classification
- [ ] Tempo classification
- [ ] Deep Learning Jazz generator
- [ ] Sort music by genres, artist, tempo, tags
- [ ] Piano analysis

-------------
## Comission Note:

#### Sep 9, 2019:
- model.h5 file is a weight of 1D Convolutional network with shape (max_feature, max_channels). Feature train: stft.
- model2D.h5 file is a weight of 2D Convolutional network with shape (max_feature, time_length, max_channels). Did not uploaded due to large file.
- model2D_2.h5 file is also a weight of 2D Convolution network, with the same shape but the channel is set in last instead of front.
- Conclusion: Feature engineering is not good due to low accuracy, more have to be done with extracting feature from data!

#### Sep 15, 2019:
- After extracting the data as spectrogram and create model based on 128x1308 pixel, I successfully train the model with an accuracy of 99.90 % and loss of 1.61%. I realize that reducing the learning rate to the factor of 1e(+n) may improve the chance of weights being rounded errorly.
- Plan for the next part: write a clear data generator for the network, write cleaner code, write documentation, seperate files by utilities, write test.

-------------


## Author: Alex Nguyen
#### Gettysburg College Class of 2022
