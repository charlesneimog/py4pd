#### py4pd version 0.8.0
* Now `py4pd` objects can have they own help-pathes, they must be located inside the folder "help".
* Add simple player embbeded (you can 'play' python objects).
* Add `pd.add2player` method.
* In `pd.addobject` add key `helppatch` (string contains the `.pd` file of help), for example, if the help is `myorchidea.flute-help` here I put `myorchidea.flute`.

#### py4pd version 0.7.0
* * Add possibility to write Python Objects (like PureData Libraries) in add to PureData as standart Objects.
* Add support to detach (It runs I separete Python executable (probably will be uncessary with PEP 684).
* Add way to work with Python Types inside PureData. It requires to send message `pointers 1`, or from Python Object, set `pyout = True`.
* Now `py4pd -library` are added in the begin of the patch, we not have problem with Py Objects not exits.
* Add new `pd` modules:
	* `getobjpointer`: It returns the string pointer to the object. Can be used to create global variables per object.
	* `iterate`: It is one copy of the OpenMusic iterate.
	* `show`: It works exactly as `pic` object, but no require the `open` message.

#### py4pd version 0.6.0
* Add audio support
  * For audioout you need to create the object with the `-audioout` flag. 
  * For audioint you need to create the object with the `-audioint` flag.
* Add vis support
  * Add support to score (using neoscore)
  * Add support to anothers visualizations (anothers like matplotlib, vispy, and others)

#### py4pd version 0.5.0.
* Add support to list inside PureData using brackts 
  * 💡 `run [1 2 3 4 5]` from `pd`message is equivalent to run `my_function([1, 2, 3, 4, 5])` in `Python`.
* Add better README and Wiki.
* Add support to new Editor [sublime, nvim, code, vim]

#### py4pd version 0.4.0
* 🤖 Add Github Actions for Windows, Linux, MacOS (Intel);
* 🛠️ Format the code;

#### py4pd version 0.3.0
* add list support (Python to PureData);
* add reload support;
* add better error messages;
* Add embedded module `pd` (just print module work for now);
* Open vscode from puredata;
* Remove thread for now;

#### py4pd version 0.2.0
* ⚠️`Incompatible versions`, now the functions are always in memory;
* Add support to Linux;
* Set functions;
* Try to add support to threads;
* First build for MacOS;
* 🛠️ Format the code;


#### py4pd version 0.1.0

* Possible to run code (without libraries);
* Save Function in Memory;
* Create scripts from PureData;
* Two ways to run Python (Clearing memory or making compiled code ready);

#### py4pd version 0.0.0

* First simple build for Windows;
