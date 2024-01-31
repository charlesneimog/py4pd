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
        "py4pd-score": {
            "link": "charlesneimog/py4pd-score",
            "description": "Simple Scores in PureData",
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
        library = py4pd_libraries[py4pdName]
        gitLink = "https://api.github.com/repos/{}/releases".format(library)
        installFolder = pd.get_pd_search_paths()[0]
        libraryPath = installFolder + "/" + py4pdName
        try:
            response = requests.get(gitLink)
            responseJson = response.json()
            sourceCodeLink = responseJson[0]["zipball_url"]
            response = requests.get(sourceCodeLink)  # download
        except Exception as e:
            pd.print("Was not possible to download the library '{}'".format(py4pdName))
            pd.error(str(e))
            return

        libraryPath = pd.get_pd_search_paths()[0] + "/" + py4pdName
        with open(libraryPath + ".zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(libraryPath + ".zip", "r") as zip_ref:
            zip_ref.extractall(pd.get_pd_search_paths()[0])
            extractFolderName = zip_ref.namelist()[0]
            os.rename(installFolder + "/" + extractFolderName, libraryPath)

        os.remove(libraryPath + ".zip")
        pd.print(f"Library '{py4pdName}' installed successfully")

except:

    def listPy4pdLibraries():
        pd.error("Requests is not installed, send 'pip install requests' to py4pd")

    def downloadPy4pdLibraries(py4pdName):
        pd.error("Request is not installed, send 'pip install requests' to py4pd")
