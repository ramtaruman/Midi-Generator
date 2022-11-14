# midi-generator

Midifile generation with Tensorflow using [Restricted Boltzmann Machine](http://deeplearning4j.org/restrictedboltzmannmachine.html).

This code implements a RNN to generate music, mostly by predicting which notes will be played at each time step of a musical piece.

To use the model, you need to obtain a hefty selection of midi music, preferably in 4/4 time signature.


## Prerquisites using PyPI
```bash
pip install tensorflow
pip install pandas
pip install msgpack
pip install numpy
pip install glob2
pip install tqdm
pip install py-midi
```

##  Midi datasets for training
There are lots of crowdsourced datasets already floating in the interwebz so I went ahead and tested some of them.

- [Classical Piano](http://www.piano-midi.de/)

- [ABC Music Project](https://abc.sourceforge.net/NMD/)

- [Lakh Dataset](https://colinraffel.com/projects/lmd/)


## Misc Reasearch Sources
There will always be some better source of literature for the given topic

- [Gibbs Chain](https://en.wikipedia.org/wiki/Gibbs_sampling)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gibbs_sampling)
- [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

## FAQ stuff
There are some common hurdles to working with the code, I will keep listing them here.

Q. Why are the weightings not seperately intialized ?

```
Conventionally, to train the model, parameters of the RBM needs to be intialized first before beginning the training of the model but this just requires more command execution in the shell. The only downside for this is the static <num_epochs> present in the code which can be initiallzed if dealing with seperate execution chain..
```
Q. How to fix "AttributeError: 'module' object has no attribute 'read_midifile'" ?
```py3
pip install git+https://github.com/vishnubob/python-midi@feature/python3
```
## Further Literature

- [Boulanger-Lewandoski 2012](https://arxiv.org/abs/1206.6392)
