The primary goal of `py4pd` was to bring the powerful tools of OpenMusic into the robust environment of PureData. This integration is considered highly significant. Similar efforts have been made in Max, with various approaches, such as the Bach Library and MozLib. In this section, we will introduce the basic functions inspired by OpenMusic and how they are emulated within PureData.

To use the objects we will present next you need to create a new patch with the object `py4pd -lib py4pd`.

### PureData and Python

One major difference you’ll notice when using Python objects with py4pd is that they typically don’t operate with standard PureData types (such as numbers, symbols, or lists). Instead, they utilize a workaround inspired by the Bach Library, using pointers. Pointers are memory addresses, and in the context of PureData and `py4pd`, they appear as references like `PyObject list <0x60610796f9d0>` or `PyObject int <0x6061079771d0>`. So in almost all `py4pd` object the output will be something like these `PyObject` thing.

At first, this may seem weird, but it's actually very powerful. Using pointers allows us to work with different data types that PureData doesn't natively support. 

For example, when performing partial tracking, we often need a data type like an **array of arrays**. Since PureData doesn't support this, being able to do Partial Tracking manipulation on PureData is hard and becomes quite difficult. 

Another scenario is when you want to build complex sound structures with many variables, parameters (like specific reverb, filter, and others configurations attached to them). PureData’s native data types can’t represent these kinds of complex objects. However, using `PyObject`, this becomes possible. 

A good demonstration of this can be seen in the `py4pd-upic` library, where sound events are represented as pointers like `PyObject SvgEvent 0x60610796f9d0>`. 

To convert Python types to PureData types you need to use the `py2pd` object (and `pd2py` for PureData to Python) but you will see that this is just necessary when you want to use the data inside some standard Pd Object.

Next we will present the set of object that comes with `py4pd`. The library is also called `py4pd`, being the default library of the `py4pd` object. To load the library you need to use `py4pd -lib py4pd`.

### List of Python Objects


Here, we present the core functions from OpenMusic along with some additional features. Some objects are still missing, this is because I implement them as I use them. If you notice any missing objects, please let us know ([click here]() to reportq) -- most can be quickly implemented!

#### <h3 align="center"><code>Logic Objects</code></h3>

Here some objects to be used when doing some logic.

<div class="grid cards" markdown>
-    ##### <h4><code>py.isnone</code></h4>
    Check if the input is `None`.
-    ##### <h4><code>py.and</code></h4>
    Check if all the inputs are `True`.
-    ##### <h4><code>py.or</code></h4>
    Check if any of the inputs are `True`.
-    ##### <h4><code>py.equal</code></h4>
    Check if two inputs are equal.
-    ##### <h4><code>py></code></h4>
    Check if the left input is greater than the right input.
-    ##### <h4><code>py<</code></h4>
    Check if the left input is less than the right input.
-    ##### <h4><code>py.if</code></h4>
    Evaluate a condition and return a result based on it.
-    ##### <h4><code>py.gate</code></h4>
    Control the flow of execution based on conditions.
-    ##### <h4><code>py.isin</code></h4>
    Check if a value exists within a collection.
</div>


--- 
#### Utils

Utilities to use `py4pd`.

<div class="grid cards" markdown>
-    ##### <h4><code>py.print</code></h4>
    Print the specified message to the console.
-    ##### <h4><code>py.raise</code></h4>
    Raise an exception (error) with the specified message.
-    ##### <h4><code>py.type</code></h4>
    Return the type of the specified object.
</div>


---
#### Convert Pd to Python | Python to Pd

Convertions between Pd and Python.

<div class="grid cards" markdown>
-    ##### <h4><code>py2pd</code></h4>
    Convert Python objects to a format compatible with Pure Data (if possible).
-    ##### <h4><code>pd2py</code></h4>
    Convert Pure Data objects to a format compatible with Python.
</div>

---
#### List Functions

Functions to work with lists.

<div class="grid cards" markdown>
-    ##### <h4><code>py.mklist</code></h4>
    Create a new list from the specified elements (require to specify the number of inputs).
-    ##### <h4><code>py.len</code></h4>
    Return the length of the specified list or collection.
-    ##### <h4><code>py.nth</code></h4>
    Retrieve the element at the specified index from a list.
-    ##### <h4><code>py.append</code></h4>
    Add an element to the end of the specified list (like `x-append` on OpenMusic).
-    ##### <h4><code>py.list</code></h4>
    Convert a given item into a list (like `list` on OpenMusic).
-    ##### <h4><code>py.split</code></h4>
    
-    ##### <h4><code>py.max</code></h4>
    Return the maximum value from a list or collection.
-    ##### <h4><code>py.min</code></h4>
    Return the minimum value from a list or collection.
-    ##### <h4><code>py.reduce</code></h4>

-    ##### <h4><code>py.mattrans</code></h4>
    Perform matrix transposition on a specified matrix (like `mattrans` from OpenMusic). 
-    ##### <h4><code>py.rotate</code></h4>
    Rotate elements of a list by a specified number of positions.
-    ##### <h4><code>py.flat</code></h4>
    Flatten a nested list into a single list (like `flat` on OpenMusic).
-    ##### <h4><code>py.np2list</code></h4>
    Convert a `numpy` array to a list.
-    ##### <h4><code>py.list2np</code></h4>
    Convert a list to a `numpy` array.
</div>


#### Loops

Function to create a Python loop inside PureData.

<div class="grid cards" markdown>
-    ##### <h4><code>py.range</code></h4>
    Generate a sequence of numbers within a specified range.
-    ##### <h4><code>py.iterate</code></h4>
    Iterate over elements in a collection.
-    ##### <h4><code>py.collect</code></h4>
    Collect results from a sequence of operations into a list.
-    ##### <h4><code>py.recursive</code></h4>
    Perform a recursive operation. It is best to check the example on `py4pd` package.
-    ##### <h4><code>py.trigger</code></h4>

</div>


#### Math

Math functions.

<div class="grid cards" markdown>
-    ##### <h4><code>py+</code></h4>
    Add two numbers or concatenate two strings.
-    ##### <h4><code>py-</code></h4>
    Subtract one number from another.
-    ##### <h4><code>py/</code></h4>
    Divide one number by another.
-    ##### <h4><code>py.abs</code></h4>
    Return the absolute value of a number.
-    ##### <h4><code>py//</code></h4>
    Perform floor division between two numbers.
-    ##### <h4><code>py.round</code></h4>
    Round a number to the nearest integer or specified decimal places.
-    ##### <h4><code>py.expr</code></h4>
    Evaluate a mathematical expression (Python list comprehension)
</div>



#### OpenMusic Objects

Objects to imitate some of the great OpenMusic objects.

<div class="grid cards" markdown>
-    ##### <h4><code>py.rhythm-tree</code></h4>
-    ##### <h4><code>py.arithm-ser</code></h4>
</div>


#### Image

Show images inside PureData.

<div class="grid cards" markdown>
-    ##### <h4><code>py.show</code></h4>
    Show image (like picture from OpenMusic).

</div>

#### Music Convertions

Convertions useful for music.

<div class="grid cards" markdown>
-    ##### <h4><code>py.f2mc</code></h4>
    Convert frequency to midicents.
-    ##### <h4><code>py.mc2f</code></h4>
    Convert midicents to frequency.
-    ##### <h4><code>py.mc2n</code></h4>
    Convert midicents to note name.
</div>

--- 

If you miss some OpenMusic object, please report to us!
