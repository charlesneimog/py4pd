import os
import zipfile

import pd

try:
    import requests
    py4pd_libraries = {
        "py4pd-upic": {
            "link": "charlesneimog/py4pd-upic",
            "description": "Use svg files as scores.",
        },
        "py4pd-ji": {
            "link": "charlesneimog/py4pd-ji",
            "description": "Just Intonation/Microtonal tools.",
        },
        "orchidea":{
            "link": "charlesneimog/orchidea",
            "description": "Get the Orchidea samples using midi data",
        },
        "py4pd-spt":{
            "link": "charlesneimog/py4pd-spt",
            "description": "Simple Partials Tracker for PureData",
        },
        "py4pd-freesound":{
            "link": "charlesneimog/py4pd-freesound",
            "description": "Search, retrieve, play, and manipulate audio samples from Freesound.org.",
        },
    }

    def listPy4pdLibraries():
        for lib in py4pd_libraries:
            pd.print(
                "=> " + str(lib) + ": " + str(py4pd_libraries[lib]["description"]),
                show_prefix=False,
            )

    def downloadPy4pdLibraries(py4pdName):
        """ """
        pd.print(f"Downloading library '{py4pdName}'")
        if py4pdName not in py4pd_libraries:
            pd.error("Library '{}' not found".format(py4pdName))
            return

        library = py4pd_libraries[py4pdName]
        gitLink = "https://api.github.com/repos/{}/releases".format(library["link"])
        installFolder = pd.get_pd_search_paths()[0]
        libraryPath = installFolder + "/" + py4pdName

        # check if file already exists
        if os.path.exists(libraryPath):
            pd.print(f"Library '{py4pdName}' already installed")
            # check if README.deken.pd exists
            if os.path.exists(libraryPath + "/README.deken.pd"):
                pd._open_patch("README.deken.pd", libraryPath)
            return

        try:
            response = requests.get(gitLink)
            responseJson = response.json()
            sourceCodeLink = responseJson[0]["zipball_url"]
            response = requests.get(sourceCodeLink)  # download
        except Exception as e:
            pd.print("The link was not found, please report using https://github.com/charlesneimog/py4pd/issues")
            pd.error(str(e))
            return

        try:
            libraryPath = pd.get_pd_search_paths()[0] + "/" + py4pdName
            with open(libraryPath + ".zip", "wb") as f:
                f.write(response.content)
        except Exception as e:
            pd.print("Error to download the library '{}'".format(py4pdName))
            pd.error(str(e))
            return

        with zipfile.ZipFile(libraryPath + ".zip", "r") as zip_ref:
            zip_ref.extractall(pd.get_pd_search_paths()[0])
            extractFolderName = zip_ref.namelist()[0]
            os.rename(installFolder + "/" + extractFolderName, libraryPath)

        thereIsRequirements = False
        if os.path.exists(libraryPath + "/requirements.txt"):
            thereIsRequirements = True
            pd._pipinstall_requirements(libraryPath + "/requirements.txt")

        if os.path.exists(libraryPath + "/README.deken.pd"):
            pd._open_patch("README.deken.pd", libraryPath)
        try:
            os.remove(libraryPath + ".zip")
        except Exception as e:
            pd.error(str(e))

        if thereIsRequirements:
            pd.error(f"Library '{py4pdName}' installed successfully, wait for the requirements to be installed")
        else:
            pd.print(f"Library '{py4pdName}' installed successfully")

except:
    def listPy4pdLibraries():
        pd.error("Requests is not installed, send 'pip install requests' to py4pd")

    def downloadPy4pdLibraries(py4pdName):
        pd.error("Request is not installed, send 'pip install requests' to py4pd")
