---
hide:
  - navigation
  - toc
---

# Installation

## <h2 align="center"> **Setup** </h2>

!!! danger "Python installation is required!"
Install just `py4pd` (no Python) will not work.

 <p align="center"> The process of installation is simple, first we install <code>Python</code> and then <code>py4pd</code>. </p>

---

### <h3 align="center"> **1. Install Python** </h3>

---

<!-- <p align="center"> -->
<!--     <iframe style="border-radius: 5px" width="560" height="315" src="https://www.youtube.com/embed/TVcHzLCmpDM?si=GIkYPPifzjhfFZvM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe> -->
<!-- </p> -->

??? danger "Check the Python Version"

     <table class="special-table" style="width: 50%" align="center">
        <thead>
          <tr>
            <th>Py4pd Version</th>
            <th>Python Version</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>0.8.0</code></td>
            <td>Python 3.11</td>
          </tr>
        </tbody>
        <tbody>
          <tr>
            <td><code>0.7.0</code></td>
            <td>Python 3.10</td>
          </tr>
        </tbody>
    </table>




=== ":fontawesome-brands-windows: Windows"

    On `Windows` you can install Python like and ordirary software.

    * Go to [Python.org](https://www.python.org/downloads/release/python-3115/),
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

    * Go to [Python.org](https://www.python.org/downloads/release/python-3115/),
    * Go to the bottom of the page and download: `macOS 64-bit universal2 installer`.
    * Install it as an ordinary program.

=== ":material-linux: Linux"

    On `Linux` you can install Python like and ordirary software.

    * Open the terminal.
    * Run: `sudo dnf install python3.11 python3-pip` (Fedora) or `sudo apt install python3.11 python3-pip` (Ubuntu) or `sudo pacman -S python3.11 python3.pip` (Arch).

### <h3 align="center"> **2. Install `py4pd`** </h3>

1. Open PureData,
2. Go to `Help->Find Externals->`,
3. Search for `py4pd`,
4. Select py4pd and click on `Install`:

<figure markdown>
  ![Install py4pd](assets/install-py4pd.gif){ width="700" loading="lazy"  style="border-radius: 3px; box-shadow: 0px 8px 8px rgba(0, 0, 0, 0.2);"}
  <figcaption>Process of install py4pd on PureData</figcaption>
</figure>
