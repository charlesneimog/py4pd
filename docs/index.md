---
hide:
  - navigation
  - toc
---

# Introduction

<p align="center"> Welcome to the <code>py4pd</code> documentation! </p>

## <h2 align="center"> **What is py4pd?** </h2>

----------

`py4pd` integrates the power of [Python](https://www.python.org/) into the real-time environment of [PureData](https://puredata.info/). Besides providing means for score visualization, AI integration, audio analysis tools, SVG/drawing score handling, and various other functionalities, <i>you can create PureData Objects using just Python</i>. If you use [OpenMusic](https://openmusic-project.github.io/openmusic/), you will find some inspiration on it.



!!! info "How to install?"

    To check how to install go to [Installation](https://www.charlesneimog.com/py4pd/setup/).

<hr>

## <h2 align="center"> **Examples and Use Cases** </h2>

Here is presented algorithm tools implemented with `py4pd`. Users are encouraged to contribute their own examples through [Github](https://github.com/charlesneimog/py4pd/discussions/categories/show-and-tell).

=== "Score In PureData"

    When I start to work with PureData, I miss a lot some score visualization tool, this can be solved by utilizing `py4pd`. This script is delivered along with the `py4pd` library.

    <p align="center">
        <img src="examples/score/score.gif" width="50%" loading="lazy" alt="Score GIF">
    </p>


=== "Artificial Inteligence"

    It is easy to use `AI` with `py4pd`. There are already powerful objects for realtime, like `nn~` and `ddsp~`, but they are designed to specify approaches. Below is an example using `py4pd` and the Python code used by `nn~` (offline processing).

    <p align="center">
	    <img src="examples/ia/ia.png" width="60%"></img>
    </p>  
    
    <p align="center">
        <audio controls style="width: 60%; border-radius: 10px;">
            <source src="examples/ia/turvo-wheel.wav" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </p>
    
    ??? example end "Python Code"

        To illustrate the statement "Python offers a more accessible and user-friendly alternative that C and C++", presented earlier, here is an example of Python code: 
        
	    ``` py 

	    import pd # py4pd library
	    import torch # Library of AI
	    import librosa # Library to load audios in Python

	    def renderAudio_nn(audio, model):
	        model = pd.home() + '/' + model # get the pathname of model.ts, that is the result of the IA trained.
	        audio = pd.home() + '/' +  audio # The audio source
	        torch.set_grad_enabled(False) # config of the IA
	        model = torch.jit.load(model).eval() # Load model of IA
	        x = librosa.load(audio)[0] # take the audio samples of the sound (audio)
	        x_for = torch.from_numpy(x).reshape(1, 1, -1) # transform the audio to fit in the IA model
	        z = model.encode(x_for) # tranlate for the IA thing, I believe here is the black box.
	        z[:, 0] += torch.linspace(-2, 2, z.shape[-1]) # No idea;
	        y = model.decode(z).numpy().reshape(-1) # Now we have sound again!
	        pd.tabwrite('iaAudio', y.tolist(), resize=True) # Here we write the sound in the table 'iaAudio'.
	        pd.print('Audio rendered')

	    ```


=== "Draws as scores"

    In this example, I use the SVG file above to render sounds using the new `else/plaits~`. Besides `earplug~`, and `cyclone/coll`. The colors control the `plaits~` parameters.


    <p align="center">
	    <img src="examples/img2sound/img2sound.jpeg"></img>
    </p>

    <p align="center">
        <audio controls style="width: 60%; border-radius: 10px;">
            <source src="examples/img2sound/img2sound.mp3" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </p>

=== "Spectral analysis"
    In Python, there is not just one Spectral Analysis package. I mainly use `loristrck` because of the `.sdif` files. But there is `simpl`, `librosa`, [PyAudio_FFT](https://github.com/aiXander/Realtime_PyAudio_FFT), among others. If you want to spectral manipulations you can work with `pysdif3` that is fast and amazing. Here an example using `loristrck` in PureData.

    <p align="center">
	    <img src="examples/spectral-analysis/analisys.gif" width="50%"></img>
    </p>

=== "Audio Descriptors Graphs"

    You can use some of the amazing Audio Descriptors provided by `audioflux` for some analisys. 

    <p align="center">
	    <img src="examples/descriptors/descriptors.png" width="50%"></img>
    </p>

----------

### <h3 align="center"> **Pieces** </h3>

=== "Eco (2023)"

    Eco (2023) is the first version of one under developing piece that use some concepts of the composer Ricardo Thomasi in his PhD research. The idea here, is to use smartphones/tablets putted in the performance music stand, to make realtime `FFT` and `Partial Tracking` and then, generate scores that are played. The smartphones/tablets send data to PureData, and using `py4pd`, we generate realtime scores using `neoscore`.

    <p align="center">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/XIEI7-W7t2o" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </p>


=== "Moteto (2023)"

    The piece Moteto aims to be a rereading of medieval motet. In addition, to overlapping independent voices, we use Just Intonation structures. With the help of technology, I aim to make the piece playable, also by amateur choirs, it is available in this website: www.charlesneimog.com/moteto/.

-------------
## <h2 align="center"> **News** </h2>
-------------

!!! info "Check the news for v. 0.8.0"
    <h3 align="center"> **<b>v. 0.8.0</b>** </h3> 
    
    * Now `py4pd` objects can have they own help-patches, they must be located inside the folder "help".
    * Add simple player embbeded (you can 'play' python objects) in the objects, in others words, put Python Objects in music time.
    * Add `pd.add2player` method.
    * Add new option `objimage` in `pd.addobject` to set default images for `VIS` objects.
    * Remove our own Python executable.
    * In `pd.addobject` add key `helppatch` (string contains the `.pd` file of help), for example, if the help is `myorchidea.flute-help` here I put `myorchidea.flute`.
    * Add new `pd` modules methods.
        * Add method `pd.clearplayer` to clear the player.
        * Add method `pd.patchzoom` to get the zoom of patch.
        * Add method `pd.pipinstall` to install Python Modules.
    * The options `py4pd -score`, `py4pd -audioin`, `py4pd -audioout`, and `py4pd -audio` was removed because they are unused when we can create your library. ⚠️
    * Added multichannel support for `audio` objects.
        
    ----------------------


??? info "Check the news for old versions"

    <h3 align="center"> **<b>v. 0.7.0</b>** </h3>
    
    * Add possibility to write Python Objects (like PureData Libraries) in add to PureData as standart Objects.
    * Add support to detach (It runs I separete Python executable (probably will be uncessary with [PEP 684](https://peps.python.org/pep-0684/)).
    * Add way to work with Python Types inside PureData. It requires to send message `pointers 1`, or from Python Object, set `pyout = True`.
    * Now `py4pd -library` are added in the begin of the patch, we not have problem with Py Objects not exits.
    * Add new `pd` modules:
	    * `getobjpointer`: It returns the string pointer to the object. Can be used to create global variables per object.
	    * `iterate`: It is one copy of the OpenMusic iterate.
	    * `show`: It works exactly as `pic` object, but no require the `open` message.

    ----------------------
    <h3 align="center"> **<b>v. 0.6.0</b>** </h3>
    
    * Add `audio` support.
      * For audioout you need to create the object with the `-audioout` flag. 
      * For audioint you need to create the object with the `-audioint` flag.
    * Add `vis` support.
      * Add support to score (using neoscore).
      * Add support to anothers visualizations (anothers like matplotlib, vispy, and others).
    * Create this beautil docs website :).

    ----------------------
    <h3 align="center"> **<b>v. 0.5.0</b>** </h3>
    
    * Add support to list inside PureData using brackts.
      * 💡 `run [1 2 3 4 5]` from `pd`message is equivalent to run `my_function([1, 2, 3, 4, 5])` in `Python`.
    * Add better README and Wiki.
    * Add support to new Editor [sublime, nvim, code, vim].

    ----------------------
    <h3 align="center"> **<b>v. 0.4.0</b>** </h3>
    
    * 🤖 Add Github Actions for Windows, Linux, MacOS (Intel);
    * 🛠️ Format the code and lot of fixes related with memory leaks.

    ----------------------
    <h3 align="center"> **<b>v. 0.3.0</b>** </h3>
    
    * add list support (Python to PureData);
    * add reload support;
    * add better error messages;
    * Add embedded module `pd` (just print module work for now);
    * Open vscode from puredata;
    * Remove thread for now;

    ----------------------
    <h3 align="center"> **<b>v. 0.2.0</b>** </h3>
    
    * ⚠️`Incompatible versions`, now the functions are always in memory;
    * Add support to Linux;
    * Set functions;
    * Try to add support to threads;
    * First build for MacOS;
    * 🛠️ Format the code;

    ----------------------
    <h3 align="center"> **<b>v. 0.1.0</b>** </h3>

    * Possible to run code (without libraries);
    * Save Function in Memory;
    * Create scripts from PureData;
    * Two ways to run Python (Clearing memory or making compiled code ready);

    ----------------------
    <h3 align="center"> **<b>v. 0.0.0</b>** </h3>

    * First simple build for Windows;


