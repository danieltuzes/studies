# studies

New fields I learn or give presentations about.

- [studies](#studies)
  - [random](#random)
  - [random test](#random-test)
  - [named random number engine](#named-random-number-engine)
  - [random-meta](#random-meta)
    - [Workflow](#workflow)
    - [OS-realted user privileges](#os-realted-user-privileges)
    - [Software requirements](#software-requirements)
      - [VS Code for development environment](#vs-code-for-development-environment)
      - [miniconda](#miniconda)
      - [Miktex](#miktex)
      - [revealjs](#revealjs)
    - [Create html](#create-html)
    - [Create pptx](#create-pptx)
    - [Create pdf with latex](#create-pdf-with-latex)

## random

This is a presentation material and also an experiment how to create a presentation in markdown and convert it to powerpoint, html and pdf for which I will use pandoc, the revealjs JS library and latex (from the Miktex distribution) and the beamer template.

This material represents a short, 45 min presentation on random numbers for those who knows a little about programming and mathematics.

To view the results as of 20th of Febr, 2021, check the [random.pdf](random.pdf), [random.pptx](random.pptx) or [random.html](random.html) files!

Some small [python examples](python/numpy_random.py) using random numbers with a random walk implementation can be found in the python folder. To test the random number generator, [3 self-written test](python/random_test.py) are also developed. These tests just shows some basic ideas and don't represent a deep analysis, testing a prng is a whole science!

## random test

Although [standardized tests for prngs](https://en.wikipedia.org/wiki/TestU01) have been introduced [decades ago](https://en.wikipedia.org/wiki/Diehard_tests), it can be still interesting to implement some of your own ideas. Most propably, those have been already implemented, but it is good to practice programming and get a deeper insight into random number generators.

In [random_test.py](python/random_test.py), you can find some (around 4) tests to check the quality of the Mersenne Twister prng. I was curious if any correlation can be found within the prn sequences initiated with adjacent seeds. Spoiler alert: I didn't find any correlation.

## named random number engine

In one of my professional tasks, I had to implement a prng container for Monte Carlo simulations. The purpose of this container is to improve prn generation by supporting multiple seeds, and instead of seeds, to support named seeds.

- The first step is to print out all the random numbers used by the model, which produced the result A, store them in a file,
- and then use this super container to feed the model with the same numbers. This time I should get the same result A.
- Then the file-based random number feed has to replaced by a prng algorithm, like Mersenne Twister. This time the model should produce different result B. I can write out the random numbers again, into a file, and use the file instead of the algorithm producing the same B result. So we can see how changing the random number engine from a basic math algorithm to file-based solution keeps the same result A.
- Then we change the random numbers, and get result B, but the result remains the same if we change the source of random numbers from a file to a math algorithm again.

The details of the class can be found in the [readme of randuti](randuti/README.md). The class is in a module `named_prng` that is distributed as a package called `randuti`.

## random-meta

Here I summarize how I created the content of the presentation.

### Workflow

1. I created the content first, with markdown, and was constantly checking it out in the preview window.
2. Once the content was more or less ready, I compiled the presentation with latex, using the beamer template, into a pdf. I added the new slide separator into the source code this time. I had problems formatting figures, because as in general, text don't flow around figures in latex so I restricted the formattings to set the width of the pictures.
3. Converted the markdown into pptx. To fit all the content into their slides and to format the slideshow, I created a `.potx` template file which mimics one of the beamer templates. The final solution is not perfect, and I didn't figure out how to correct the pictures in the slide show to not to be placed in a new slide and force another slide-break before the next content.
4. Converted the markdown into html using revealjs. To fit the content onto the slides, I created a new style `simple2.css`.

### OS-realted user privileges

In a corporate environemnt, it is important to be aware of software limitations and I will keep this in mind during the presentation development. All software should be able to installed without admin privileges, or are so basic all corporates should allow them to require.

### Software requirements

This is not the only set of software which makes it possible to create the presentations, but one way to achieve it.

- VS Code: the development environment
- miniconda: this will help to install pandoc, the converter between markdown and all other languages like power point, latex and html
- Miktex: the latex distribution. This creates the pdf from the latex file generated by pandoc.
- revealjs: this is needed for pandoc and for the html to create and display the slideshow in html
- power point: if you want to view the `pptx` output, you need this one too
- git: optional, with this, you can add revealjs as a submodule and refresh it

#### VS Code for development environment

I use VS Code for development, because

- it supports all the languages I am working with: C++, python, Markdown, html, jinja, latex, vba, shell
- has integrated git support, however, it is not the best in the class, PyCharm's built in git manager seems to be better, but GitHub Desktop, a git manager not only for github repos, is maybe the best
- small, fast and highly customizable. It is possible to preview pdfs, markdowns, and many other formats

Once you have it, I suggest to install the following extensions at least:

- `ms-python.python` package for managing python and anaconda
- `yzhang.markdown-all-in-one` to edit markdown easier

#### miniconda

To install pandoc with conda, you need miniconda. You can also install pandoc individually, but if you already have conda, then there is no need to get pandoc individually.

You can also install miniconda yourself, after downloading it from the internet, but I'll suppose you already have it.

Open an anaconda prompt and create a new environment, where you have the most modern pandoc:

```PowerShell
(base) C:\Users\tuzes>conda create --name pandoc pandoc
(base) C:\Users\tuzes>conda activate pandoc
(pandoc) C:\Users\tuzes>
```

Here we installed pandoc with conda, and activated it. Whenever you need pandoc, you will need to start a command prompt that knows where conda is and activate an environment where pandoc can be found. You can also set up VS Code so that when you start a new terminal, it starts a kind of an Anaconda prompt that knows where conda is therefore you can activate the pandoc environment.

#### Miktex

Download and install Miktex, and update the packages as well. If you don't have admin rights, you can still install packages into your home directory, but most probably Miktex package manager and installer is not allowed to reach the internet from a corporate environment, in which case you need to download tha packages manually and install them, or download the files and place them right next to your generated latex files, such as `beamer.cls`, evershi.sty and `l3backend-padftex.def`.

It may happen that you don't have pdflatex in your VS Code path, but you have it in your Anaconda prompt's path. Although I don't understand why it happens, you can still add the path to your pdflatex into your integrated PowerShell terminal. Edit the `settings.json` file in your `.vscode` folder (Ctrl+Shift+P, then type Open Workspace Settings (JSON)) and add the following lines to the beginning of the file

```JSON
{
    "terminal.integrated.shellArgs.windows": [
        "-NoExit",
        "$Env:Path += ';C:\\Users\\tuzes\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\'"
    ],
    "markdownlint.config": {
        "MD025": false
    },

}
```

Modify the path to your pdflatex in the script.

#### revealjs

You need this couple of JS and CSS files for the html output to work properly. To be more precise, your software will generate the CSS file from SCSS files you download with revealjs.

- Without version control

  If you don't version track your repo or don't want to add [revealjs](https://revealjs.com/installation/) as a submodule, then [download it from github as a zip](https://github.com/hakimel/reveal.js/archive/master.zip), and unpack it as a folder called `reveal.js` next to your presentation file `random.md`.

  I also included a new theme called `simple2.css`, placed in the root, which has to be copied to `\reveal.js\dist\theme\simple2.css"`. This has a smaller font-size so that all the text aimed to fit into a slide fit into a slide. (I used the beamer template from latex, with default settings, to decide how much content I should put onto one slide.)

- If you have git and want to add revealjs as a submodule, you can
  - add the original [revealjs](https://revealjs.com/installation/)
    ```bash
    tuzes@PCname MINGW64 ~/source/repos/studies (main)
    $ git submodule add https://github.com/hakimel/reveal.js.git
    ```
    
    You also need to manually copy [`simple2.css`](https://github.com/danieltuzes/reveal.js/blob/smaller-font/dist/theme/simple2.css) to `\reveal.js\dist\theme\simple2.css`.
    
  - insted of the original version, you can use my fork if it with the extra style file:

    ```bash
    tuzes@PCname MINGW64 ~/source/repos/studies (main)
    $ git submodule add https://github.com/danieltuzes/reveal.js
    ```

    Then checkout the branch `smaller-font` within the submodule,

    ```bash
    tuzes@PCname MINGW64 ~/source/repos/studies (main)
    $ cd reveal.js/

    tuzes@PCname MINGW64 ~/source/repos/studies/reveal.js ((c27e3b5...))
    $ git checkout smaller-font
    Previous HEAD position was c27e3b5 add data-auto-animate-id to auto-animate examples #2896
    Switched to a new branch 'smaller-font'
    Branch 'smaller-font' set up to track remote branch 'smaller-font' from 'origin'.
    ```

    Now you have my branch with my modifications.

### Create html

Activate your python environment where you have `pandoc` installed and issue

```PowerShell
(pandoc) PS C:\Users\tuzes\source\repos\studies> pandoc -t revealjs -o random.html --self-contained -V revealjs-url=reveal.js --mathjax random.md -V theme=simple2
```

This tells pandoc to

- [target](https://pandoc.org/MANUAL.html#option--to) revealjs,
- the [output file](https://pandoc.org/MANUAL.html#option--output) is `random.html`,
- it should be [self-contained](https://pandoc.org/MANUAL.html#option--self-contained), i.e. don't depend on external files (like pictures, scripts and styles), but inclue them all in the file,
- to use the folder `reveal.js` as the source of the revealjs instead of `https://unpkg.com/reveal.js@^4/` to which you may have no access
- [Use mathjax (from cloudfare)](https://pandoc.org/MANUAL.html#option--mathjax) to display math equations
- [use input file](https://pandoc.org/MANUAL.html#synopsis) `random.md`
- use the settings file I created, don't forget to [copy the simple2.css](#copy-the-simple2css) into the themes!

Pandoc will then generate `random.html` within 10 s. Open it with Firefox and use the arrow keys for navigation. pandoc may need to connect to the internet to download fonts.

### Create pptx

Issue

```PowerShell
(pandoc) PS C:\Users\tuzes\source\repos\studies> pandoc -o random.pptx random.md --reference-doc '.\Beamer_template.potx'
```

This will create a power point presentation file that can be opened with power point 2013 or newer. I also provided a template to format the presentation.

### Create pdf with latex

Issue

```PowerShell
(pandoc) PS C:\Users\tuzes\source\repos\studies> pandoc -t beamer -o random.pdf random.md
```

This tells pandoc to

- [target](https://pandoc.org/MANUAL.html#option--to) beamer, a presentation template from latex,
- the [output file](https://pandoc.org/MANUAL.html#option--output) is `random.pdf`, this will be created by pdflatex from an intermediate, temporal latex file
- from the input file `random.md`

If you don't have the latex packages to compile the temporal latex file, you will need to install them. These files (from these packages) are needed:

- beamer.cls (beamer)
- geometry.sty (geometry)
- ltxcmds.sty (ltxcmds)
- infwarerr.sty (infwarerr)
- kvsetkeys.sty (kvsetkeys)
- kvdefinekeys.sty (kvdefinekeys)
- pdfescape.sty (pdfescape)
- hycolor.sty (hycolor)
- letltxmacro.sty (letltxmacro)
- auxhook.sty (auxhook)
- kvoptions.sty (kvoptions)
- intcalc.sty (intcalc)
- etexcmds.sty (etexcmds)
- bitset.sty (bitset)
- bigintcalc.sty (bigintcalc)
- rerunfilecheck.sty (rerunfilecheck)
- uniquecounter.sty (uniquecounter)
- sansmathaccent.sty (sansmathaccent)
- etc...

You may need to add a proxy to your miktex pacakge manager and try to choose a good repository. If you cannot you need to download the files manually, and either set up a local repository server or copy the requested file next to the .md file.
