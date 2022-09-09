# VirtuosoNet

Our research project is developing a system for generating expressive piano perfomrance, or simply 'AI Pianist'. The system reads a given music score in MusicX
ML and generates a human-like performance MIDI file.

This repository contains PyTorch code and pre-trained model for [__Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance__ (ICML 2019)](http://proceedings.mlr.press/v97/jeong19a.html), and [__VirtuosoNet: A Hierarchical RNN-based system for modeling expressive piano performance__ (ISMIR 2019)](http://archives.ismir.net/ismir2019/paper/000112.pdf).

This documentation is currently a work in progress.
contact: jdasam@kaist.ac.kr

* You need PyTorch (torch 0.4.1) and pretty_midi (https://github.com/craffel/pretty-midi)
* Init submodule
> git submodule update --init --recursive



## How to generate performance MIDI from musicXML

0. Put your musicXML in a folder. 
The filename shouldbe 'musicxml_cleaned.musicxml' or 'xml.xml' or 'musicxml.musicxml'
We recommend ./test_pieces/

1. Select the composer of piece.
There are 16 composers in our data set: 'Bach', 'Balakirev', 'Beethoven', 'Brahms', 'Chopin', 'Debussy', 'Glinka', 'Haydn', 'Liszt', 'Mozart', 'Prokofiev', 'Rachmaninoff', 'Ravel', 'Schubert', 'Schumann', 'Scriabin'
You can select one of them using -comp=<composer name>. The input composer does not have to be the same with the actual composer of the input piece. 
We recommend to use composer among the following list because they have more data than others: Bach, Beethoven, Chopin, Haydn, Liszt, Mozart, Ravel, Schubert
(example -comp=Mozart) (start with capital letter)

2. select model
model code is: isgn (proposed in ICML2019), han_ar_single(proposed in ISMIR 2019), han_ar_note(proposed in ISMIR 2019)
(example: -code=isgn)

* (Option) select initial tempo
You can select initial tempo of the piece in quater note per minute. If you do not enter tempo, the tempo used in MusicXML file will be used.

3. run python script
> python3 model_run.py -mode=test -code=isgn -path=./test_pieces/bwv_858_prelude/ -comp=Bach -tempo=60)

4. You can use -mode=testAll to generate performance for the pre-defined test set, which is defined in model_constants.py
It will encode emotion cue from pre-recorded performances in emotionNet folder, and generate the performance with encoded z for each emotion for each piece in the list. 'OR' represent original, or natural emotion of the piece.
> python3 model_run.py -mode=testAll -code=isgn

You can also generate performance for the pre-defined test set only 


5. check the output file
The file is saved in ./test_result/ folder. z0 means latent vector z was sampled from normal distribution.



* Caution on pedal
We add sustain pedal and soft pedal in MIDI file as a CC event of channel 64 and 67. Depending on your MIDI player, the pedal can be applied in different way. For example, Logic Pro X activate pedal if the value is lager than zero, while the actual Disklavier's pedal threshold is about 64. In this case, our performance will sound too 'wet', or too much pedal. In this case, we propose to use option -bp=true (--boolPedal), which makes value of pedal event zero under certain threshold.

If the MIDI player cannot handle pedal, the articulation of our notes will sound extremly short, since the performance we used for training set did not consider much to the articulation of notes with pedal.



## How to train the model
You can change model parameters in model_parameters.py
> python3 model_run.py -mode=train -code=isgn_test -data=training_data)





