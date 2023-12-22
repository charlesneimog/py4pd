## `pd.show_image`

With `py4pd` you can display images inside PureData patches using Python, you can use the `pd.show_image` method. This method is essentially a copy of the `else/pic` object, but with an interface that allows you to easily show images from within your Python code.

!!! warning "Supported extensions"
    You can just use `.png`, `.gif`, and `.ppm` image formats. 



<div class="grid cards" markdown>

-   :fontawesome-brands-python: __Example__


    Bring scores to PureData.

    ??? code Python Code
        ``` python
        import pd
        from random import randint
        import os
        try:
            from neoscore.common import *
        except Exception as e:
            pd.error(str(e))
            pd.error(
                "Please, run 'pip install neoscore -t ./py-modules' in the terminal from current folder")


        def getpitchKey(pitch):
            note = {
                # natural
                'c': ['c', ''],
                'd': ['d', ''],
                'e': ['e', ''],
                'f': ['f', ''],
                'g': ['g', ''],
                'a': ['a', ''],
                'b': ['b', ''],
                # sharp
                'c#': ['c', 'accidentalSharp'],
                'd#': ['d', 'accidentalSharp'],
                'e#': ['e', 'accidentalSharp'],
                'f#': ['f', 'accidentalSharp'],
                'g#': ['g', 'accidentalSharp'],
                'a#': ['a', 'accidentalSharp'],
                'b#': ['b', 'accidentalSharp'],
                # flat
                'cb': ['c', 'accidentalFlat'],
                'db': ['d', 'accidentalFlat'],
                'eb': ['e', 'accidentalFlat'],
                'fb': ['f', 'accidentalFlat'],
                'gb': ['g', 'accidentalFlat'],
                'ab': ['a', 'accidentalFlat'],
                'bb': ['b', 'accidentalFlat'],
            }
            return note[pitch]


        def chord(pitches):
            try:
                neoscore.shutdown()
            except BaseException:
                pass
            neoscore.setup()
            py4pdTMPfolder = pd.get_temp_dir()
            for file in py4pdTMPfolder:
                if file.endswith(".ppm"):
                    try:
                        os.remove(py4pdTMPfolder + "/" + file)
                    except BaseException:
                        pass
            staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
            trebleClef = 'treble'
            Clef(ZERO, staffSoprano, trebleClef)
            staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
            bassClef = 'bass'
            Clef(ZERO, staffBaixo, bassClef)
            Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42),
                      Brush(Color(0, 0, 0, 0)), Pen(thickness=Mm(0.5)))
            for pitch in pitches:
                # in pitch remove not number
                pitchWithoutNumber = pitch.replace(pitch[-1], '')
                pitchOctave = int(pitch[-1])
                pitchClass, accidental = getpitchKey(pitchWithoutNumber)
                note = [(pitchClass, accidental, pitchOctave)]
                if pitchOctave < 4:
                    Chordrest(Mm(5), staffBaixo, note, (int(1), int(1)))
                else:
                    Chordrest(Mm(5), staffSoprano, note, (int(1), int(1)))
            randomNumber = randint(1, 100)
            notePathName = py4pdTMPfolder + "/" + pitch + f"{randomNumber}.ppm"
            neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
            neoscore.shutdown()
            if os.name == 'nt':
                notePathName = notePathName.replace("\\", "/")
            pd.show_image(notePathName) ## HERE THE pd.show_image
            return None

        ```
    <p align="center">
      <img src="../../examples/score/score.gif" alt="pd.out Example" width="100%">
    </p>



-   :fontawesome-solid-gear: __Arguments__

    | Parameters     | Type | Description                   | 
    | :-----------: | :----: | :------------------------------: |
    | `arg1`   | `string` | Pathname for the image that will be showed. |

</div>

---

## `pd.get_pxsize` 


TODO.
