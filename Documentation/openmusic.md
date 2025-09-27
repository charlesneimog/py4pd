# Algorithm Composition

`py4pd` offers a list of objects to create algorithm composition, they mimic some obcjets of [OpenMusic](https://openmusic-project.github.io/openmusic/).


!!! warning "I am yet implementing most of OpenMusic objects"


## Visual Language


### Lisp

Basic functions defined in the programming language

* `py.first`: Returns the first element of a list.
* `py.second`: Returns the second element of a list.
* `py.third`: Returns the third element of a list.
* `py.nth`: Returns the element at a specified index in a list.
* `py.rest`: Returns all elements of a list except the first.
* `py.nthcdr`: Returns all elements of a list starting from a specified index.
* `py.butlast`: Returns all elements of a list except the last.
* `py.reverse`: Returns a new list with elements in reverse order.
* `py.length`: Returns the number of elements in a list.
* `py.list`: Converts arguments into a list.
* `py.remove`: Returns a list with all occurrences of a given element removed.
* `py.cons`: Constructs a new list by prepending an element to an existing list.
* `py.append`: Concatenates two lists into a single list.
* `py.apply`: Calls a function with a list of arguments.
* `py.funcall`: Calls a function with given arguments directly.
* `py.mapcar`: Applies a function to each element of a list and returns a list of results.
* `py.mapcan`: Applies a function to each element of a list and concatenates all results into a single list.


### Control

Special boxes implementing control operators, argument passing and memory

* `py.seq`:
* `py.hub`:
* `py.split`:
* `py.default`:

### Loop

Special boxes for visual loop implementation

Special symbols:

* `py.iterate`: Iterate over a list.
* `py.collect`: Collect items. Must work with `py.iterate`.
<!-- * `py.tcollect`: -->
<!-- * `py.init-do`: -->
<!-- * `py.loop-for`: -->
<!-- * `py.loop-while`: -->
<!-- * `py.loop-list`: -->
<!-- * `py.loop-tail`: -->
<!-- * `py.accum`: -->

<!-- ### Files -->
<!---->
<!-- File I/O management and utilities -->
<!---->
<!-- * `py.infile`: -->
<!-- * `py.outfile`: -->
<!-- * `py.tmpfile`: -->
<!-- * `py.home-directory`: -->
<!-- * `py.folder-contents`: -->
<!-- * `py.create-pathname`: -->
<!-- * `py.pathname-directory`: -->
<!-- * `py.pathname-name`: -->
<!-- * `py.pathname-type`: -->
<!-- * `py.open-file-stream`: -->
<!-- * `py.close-file-stream`: -->
<!-- * `py.file-write`: -->
<!-- * `py.file-write-line`: -->
<!-- * `py.file-read-line`: -->

## Basic Tools

Objects and tools for data representation and processing


<!-- ### List Processing -->
<!---->
<!-- * `py.last-elem`: -->
<!-- * `py.last-n`: -->
<!-- * `py.first-n`: -->
<!-- * `py.x-append`: -->
<!-- * `py.flat`: -->
<!-- * `py.create-list`: -->
<!-- * `py.expand-lst`: -->
<!-- * `py.mat-trans`: -->
<!-- * `py.group-list`: -->
<!-- * `py.remove-dup`: -->
<!-- * `py.subs-posn`: -->
<!-- * `py.interlock`: -->
<!-- * `py.list-modulo`: -->
<!-- * `py.list-explode`: -->
<!-- * `py.list-filter`: -->
<!-- * `py.table-filter`: -->
<!-- * `py.band-filter`: -->
<!-- * `py.range-filter`: -->
<!-- * `py.posn-match`: -->


### Arithmetic

* `py.+`: + lists, number and list with number.
* `py.-`: - lists, number and list with number.
* `py.*`: * lists, number and list with number.
* `py./`: / lists, number and list with number.

<!-- * `py.om//`: -->
<!-- * `py.om^`: -->
<!-- * `py.om-e`: -->
<!-- * `py.om-abs`: -->
<!-- * `py.om-min`: -->
<!-- * `py.om-max`: -->
<!-- * `py.list-min`: -->
<!-- * `py.list-max`: -->
<!-- * `py.all-equal`: -->
<!-- * `py.om-mean`: -->
<!-- * `py.om-log`: -->
<!-- * `py.om-round`: -->
<!-- * `py.om-scale`: -->
<!-- * `py.om-scale/sum`: -->
<!-- * `py.reduce-tree`: -->
<!-- * `py.interpolation`: -->
<!-- * `py.factorize`: -->
<!-- * `py.om-random`: -->
<!-- * `py.perturbation`: -->
<!-- * `py.om<`: -->
<!-- * `py.om>`: -->
<!-- * `py.om<=`: -->
<!-- * `py.om>=`: -->
<!-- * `py.om=`: -->
<!-- * `py.om/=`: -->
<!---->
<!---->
<!-- ### Combinatorial -->
<!---->
<!-- * `py.sort-list`: -->
<!-- * `py.rotate`: -->
<!-- * `py.nth-random`: -->
<!-- * `py.permut-random`: -->
<!-- * `py.posn-order`: -->
<!-- * `py.permutations`: -->
<!---->
<!---->
<!-- ### Series -->
<!---->
<!-- * `py.arithm-ser`: -->
<!-- * `py.geometric-ser`: -->
<!-- * `py.fibo-ser`: -->
<!-- * `py.inharm-ser`: -->
<!-- * `py.prime-ser`: -->
<!-- * `py.prime?`: -->
<!-- * `py.x->dx`: -->
<!-- * `py.dx->x`: -->
<!---->
<!---->
<!-- ### Sets -->
<!---->
<!-- * `py.x-union`: -->
<!-- * `py.x-intersect`: -->
<!-- * `py.x-Xor`: -->
<!-- * `py.x-diff`: -->
<!-- * `py.included?`: -->
<!---->
<!---->
<!-- ### Interpolation -->
<!---->
<!-- * `py.x-transfer`: -->
<!-- * `py.y-transfer`: -->
<!-- * `py.om-sample`: -->
<!-- * `py.linear-fun`: -->
<!-- * `py.reduce-points`: -->
<!-- * `py.reduce-n-points`: -->
<!---->
<!---->
<!-- ### Curves & Functions -->
<!---->
<!-- * `py.point-pairs`: -->
<!-- * `py.bpf-interpol`: -->
<!-- * `py.bpf-scale`: -->
<!-- * `py.bpf-extract`: -->
<!-- * `py.bpf-offset`: -->
<!-- * `py.bpf-crossfade`: -->
<!-- * `py.bpf-spline`: -->
<!-- * `py.set-color`: -->
<!-- * `py.jitter`: -->
<!-- * `py.vibrato`: -->
<!-- * `py.param-process`: -->
<!---->
<!-- Classes: -->
<!-- * `py.bpf`: -->
<!-- * `py.bpc`: -->
<!---->
<!---->
<!-- ### Text -->
<!---->
<!-- * `py.textbuffer-eval`: -->
<!-- * `py.textbuffer-read`: -->
<!---->
<!-- Classes: -->
<!-- * `py.textbuffer`: -->
<!---->
<!---->
<!-- ### Arrays -->
<!---->
<!-- * `py.get-field`: -->
<!---->
<!-- #### Class-Array and Components -->
<!---->
<!-- * `py.new-comp`: -->
<!-- * `py.get-comp`: -->
<!-- * `py.add-comp`: -->
<!-- * `py.remove-comp`: -->
<!-- * `py.comp-list`: -->
<!-- * `py.comp-field`: -->
<!---->
<!---->
<!-- ## Score -->
<!---->
<!-- Score tools and objects -->
<!---->
<!-- Classes: -->
<!-- * `py.note`: -->
<!-- * `py.chord`: -->
<!-- * `py.chord-seq`: -->
<!-- * `py.voice`: -->
<!-- * `py.multi-seq`: -->
<!-- * `py.poly`: -->
<!---->
<!---->
<!-- ### Score Tools -->
<!---->
<!-- Manipulation of score objects -->
<!---->
<!-- * `py.object-dur`: -->
<!-- * `py.get-chords`: -->
<!-- * `py.concat`: -->
<!-- * `py.select`: -->
<!-- * `py.insert`: -->
<!-- * `py.merger`: -->
<!-- * `py.align-chords`: -->
<!-- * `py.split-voices`: -->
<!-- * `py.true-durations`: -->
<!---->
<!---->
<!-- ### Rhythm -->
<!---->
<!-- Operations on rhythm trees and ratios -->
<!---->
<!-- * `py.mktree`: -->
<!-- * `py.tree2ratio`: -->
<!-- * `py.pulsemaker`: -->
<!-- * `py.maketreegroups`: -->
<!-- * `py.n-pulses`: -->
<!-- * `py.group-pulses`: -->
<!-- * `py.get-pulse-places`: -->
<!-- * `py.get-rest-places`: -->
<!-- * `py.get-signatures`: -->
<!-- * `py.reducetree`: -->
<!-- * `py.tietree`: -->
<!-- * `py.filtertree`: -->
<!-- * `py.reversetree`: -->
<!-- * `py.rotatetree`: -->
<!-- * `py.remove-rests`: -->
<!-- * `py.subst-rhythm`: -->
<!-- * `py.invert-rhythm`: -->
<!-- * `py.omquantify`: -->
<!---->
<!---->
<!-- ### Extras / Groups -->
<!---->
<!-- Extra elements attached to chords in score editors. -->
<!---->
<!-- * `py.add-extras`: -->
<!-- * `py.remove-extras`: -->
<!-- * `py.get-extras`: -->
<!-- * `py.get-segments`: -->
<!-- * `py.map-segments`: -->
<!---->
<!-- Classes: -->
<!-- * `py.score-marker`: -->
<!-- * `py.head-extra`: -->
<!-- * `py.vel-extra`: -->
<!-- * `py.text-extra`: -->
<!-- * `py.symb-extra`: -->
<!---->
<!---->
<!-- ### Utils -->
<!---->
<!-- Unit conversion utilities etc. -->
<!---->
<!-- * `py.approx-m`: -->
<!-- * `py.mc->f`: -->
<!-- * `py.f->mc`: -->
<!-- * `py.mc->n`: -->
<!-- * `py.n->mc`: -->
<!-- * `py.int->symb`: -->
<!-- * `py.symb->int`: -->
<!-- * `py.beats->ms`: -->
<!---->
<!---->
<!-- ### Math -->
<!---->
<!-- Mathematical tools and Set theory -->
<!---->
<!-- * `py.chord2c`: -->
<!-- * `py.c2chord`: -->
<!-- * `py.c2chord-seq`: -->
<!-- * `py.chord-seq2c`: -->
<!-- * `py.c2rhythm`: -->
<!-- * `py.rhythm2c`: -->
<!-- * `py.nc-rotate`: -->
<!-- * `py.nc-complement`: -->
<!-- * `py.nc-inverse`: -->
<!---->
<!---->
<!-- ### Import / Export -->
<!---->
<!-- Import and export utilities -->
<!---->
<!-- * `py.import-musicxml`: -->
<!-- * `py.import-midi`: -->
<!-- * `py.save-as-midi`: -->
<!---->
<!---->
<!-- ## Audio -->
<!---->
<!-- Sound/DSP objects and support -->
<!---->
<!-- * `py.sound-dur`: -->
<!-- * `py.sound-dur-ms`: -->
<!-- * `py.sound-samples`: -->
<!-- * `py.save-sound`: -->
<!---->
<!-- Classes: -->
<!-- * `py.sound`: -->
<!---->
<!---->
<!-- ### Processing -->
<!---->
<!-- * `py.sound-silence`: -->
<!-- * `py.sound-cut`: -->
<!-- * `py.sound-fade`: -->
<!-- * `py.sound-mix`: -->
<!-- * `py.sound-seq`: -->
<!-- * `py.sound-normalize`: -->
<!-- * `py.sound-gain`: -->
<!-- * `py.sound-mono-to-stereo`: -->
<!-- * `py.sound-to-mono`: -->
<!-- * `py.sound-stereo-pan`: -->
<!-- * `py.sound-merge`: -->
<!-- * `py.sound-split`: -->
<!-- * `py.sound-resample`: -->
<!-- * `py.sound-loop`: -->
<!-- * `py.sound-reverse`: -->
<!---->
<!---->
<!-- ### Analysis -->
<!---->
<!-- * `py.sound-rms`: -->
<!-- * `py.sound-transients`: -->
<!---->
<!---->
<!-- ### Conversions -->
<!---->
<!-- * `py.lin->db`: -->
<!-- * `py.db->lin`: -->
<!-- * `py.samples->sec`: -->
<!-- * `py.sec->samples`: -->
<!-- * `py.ms->sec`: -->
<!-- * `py.sec->ms`: -->
<!---->
<!---->
<!-- ### Tools -->
<!---->
<!-- * `py.adsr`: -->
<!---->
<!-- ### SDIF -->
<!---->
<!-- Tools for manipulating data in the Standard Description Interchange Format (SDIF) -->
<!---->
<!-- Classes: -->
<!-- * `py.sdiffile`: -->
<!---->
<!---->
<!-- ## SDIF Structures -->
<!---->
<!-- * `py.find-in-nvt`: -->
<!-- * `py.find-in-nvtlist`: -->
<!---->
<!-- Classes: -->
<!-- * `py.sdifframe`: -->
<!-- * `py.sdifmatrix`: -->
<!-- * `py.sdiftype`: -->
<!-- * `py.sdifnvt`: -->
<!---->
<!---->
<!-- ### Read and Convert -->
<!---->
<!-- * `py.SDIFInfo`: -->
<!-- * `py.SDIFTypeDescription`: -->
<!-- * `py.GetNVTList`: -->
<!-- * `py.GetSDIFData`: -->
<!-- * `py.GetSDIFTimes`: -->
<!-- * `py.GetSDIFFrames`: -->
<!-- * `py.GetSDIFPartials`: -->
<!-- * `py.GetSDIFChords`: -->
<!-- * `py.SDIF->chord-seq`: -->
<!-- * `py.SDIF->bpf`: -->
<!-- * `py.SDIF->markers`: -->
<!-- * `py.SDIF->text`: -->
<!---->
<!---->
<!-- ### Write -->
<!---->
<!-- * `py.write-sdif-file`: -->
<!-- * `py.bpf->sdif`: -->
<!-- * `py.markers->sdif`: -->
<!-- * `py.chord-seq->sdif`: -->
<!-- * `py.open-sdif-stream`: -->
<!-- * `py.sdif-write-frame`: -->
<!-- * `py.sdif-write-header`: -->
<!---->
<!-- ## 3D -->
<!---->
<!-- Classes: -->
<!-- * `py.3DC`: -->
<!---->
<!-- * `py.3D-sample`: -->
<!-- * `py.3D-interpol`: -->
<!---->
<!---->
<!-- #### 3D-model -->
<!---->
<!-- * `py.get-transformed-data`: -->
<!---->
<!-- Classes: -->
<!-- * `py.3D-model`: -->
<!-- * `py.3D-cube`: -->
<!-- * `py.3D-sphere`: -->
<!-- * `py.3D-lines`: -->
<!---->
<!---->
<!-- #### Conversions -->
<!---->
<!-- * `py.xyz->aed`: -->
<!-- * `py.aed->xyz`: -->
<!---->
<!---->
<!-- #### Background Elements -->
<!---->
<!-- Classes: -->
<!-- * `py.speaker`: -->
<!-- * `py.project-room`: -->
<!---->
<!---->
<!-- ### Conversions (sub-package of Basic Tools) -->
<!---->
<!-- * `py.car->pol`: -->
<!-- * `py.pol->car`: -->
<!-- * `py.xy->ad`: -->
<!-- * `py.ad->xy`: -->
