# Installation

## <h2 align="center"> **Setup** </h2>

!!! danger "Python installation is required!"
	Install just `py4pd` (no Python) will not work. 

 <p align="center"> The process of installation is simple, first we install <code>Python</code> and then <code>py4pd</code>. </p>

-------------------
### <h3 align="center"> **1. Install Python** </h3>
-------------------

=== ":fontawesome-brands-windows: Windows"
    
    On `Windows` you can install Python like and ordirary software.

    * Go to [Python.org](https://www.python.org/downloads/release/python-31011/),
    * Go to the bottom of the page and download: `Windows installer (64-bit)`.
    * Install it as an ordinary program.

    !!! info "Click if you are a Windows User"
	    I recommend mark the option `Add Python 3.10 to PATH`.	
	    <figure markdown>
	      	![Install py4pd](assets/installPython.jpg){ width="500" }
	      	<figcaption>Process of install py4pd on PureData</figcaption>
	    </figure>
	    
=== ":material-apple: macOS"

    On `MacOS` you can install Python like and ordirary software.

    * Go to [Python.org](https://www.python.org/downloads/release/python-31011/),
    * Go to the bottom of the page and download: `macOS 64-bit universal2 installer`.
    * Install it as an ordinary program.
    
    
=== ":material-linux: Linux"

    On `Linux` you can install Python like and ordirary software.

    * Open the terminal.
    * Run: `sudo dnf install python3.10` (Fedora) or `sudo apt install python3.10` (Ubuntu). 
	
??? danger "Attention to Python Version"
    If you want to use old versions, the Python must follow the exact version of the compilation: 
 	
    | py4pd  version     | Python   version                            |
    | :---------: | :----------------------------------: |
    | 0.7.0       | [Python 3.10](https://www.python.org/downloads/release/python-31011/)  |
    | 0.6.0       | [Python 3.10](https://www.python.org/downloads/release/python-31010/)  |
    | 0.5.0       | [Python 3.11](https://www.python.org/downloads/release/python-3112/)  |
    | 0.4.0       | [Python 3.11](https://www.python.org/downloads/release/python-3112/)  |
    | 0.3.0       | [Python 3.10](https://www.python.org/downloads/release/python-31010/)  |
    | 0.2.0       | [Python 3.10](https://www.python.org/downloads/release/python-3105/)  |
    | 0.1.0       | [Python 3.10](https://www.python.org/downloads/release/python-3103/)  |
    | 0.0.0       | [Python 3.10](https://www.python.org/downloads/release/python-3101/)  |

------------------
### <h3 align="center"> **2. Install `py4pd`** </h3>
------------------
1. Open PureData, 
2. Go to `Help->Find Externals->`,
3. Search for `py4pd`,
4. Select py4pd and click on `Install`: 

<figure markdown>
  ![Install py4pd](assets/install-py4pd.gif){ width="700" loading="lazy"}
  <figcaption>Process of install py4pd on PureData</figcaption>
</figure>



