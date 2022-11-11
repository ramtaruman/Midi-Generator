# midi-generator

Midifile generation with Tensorflow using [Restricted Boltzmann Machine](http://deeplearning4j.org/restrictedboltzmannmachine.html).

This code implements a RNN to generate music, mostly by predicting which notes will be played at each time step of a musical piece.

To use the model, you need to obtain a hefty selection of midi music, preferably in 4/4 time signature.
